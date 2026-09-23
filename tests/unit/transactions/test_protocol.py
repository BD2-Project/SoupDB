"""Tests for the wire protocol codec (v1)."""

import socket

import pytest

from engine.common.errors import (
    DeadlockDetected,
    ProtocolError,
    QueryExecutionError,
    QueryParseError,
    TransactionError,
)
from engine.common.schema import ColumnDef, ColumnType
from engine.query.resultset import ResultSet
from engine.transactions import protocol as proto

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
)


def test_frame_roundtrip() -> None:
    payload = b"hola"
    frame = proto.encode_frame(proto.OP_QUERY, payload)
    opcode, decoded = proto.decode_frame(frame)
    assert opcode == proto.OP_QUERY
    assert decoded == payload


def test_frame_layout() -> None:
    frame = proto.encode_frame(proto.OP_BEGIN)
    assert frame[:2] == proto.MAGIC
    assert frame[2] == proto.VERSION
    assert frame[3] == proto.OP_BEGIN
    assert int.from_bytes(frame[4:8], "big") == 0


def test_bad_magic_raises() -> None:
    with pytest.raises(ProtocolError):
        proto.decode_frame(b"XX" + b"\x01\x06\x00\x00\x00\x00")


def test_unsupported_version_raises() -> None:
    frame = proto.MAGIC + bytes((0x63, proto.OP_QUERY)) + b"\x00\x00\x00\x00"
    with pytest.raises(ProtocolError):
        proto.decode_frame(frame)


def test_short_frame_raises() -> None:
    with pytest.raises(ProtocolError):
        proto.decode_frame(b"SP\x01")


def test_query_roundtrip() -> None:
    sql = "SELECT * FROM papers WHERE anio = 2020"
    assert proto.decode_query(proto.encode_query(sql)) == sql


def test_ok_roundtrip() -> None:
    assert proto.decode_ok(proto.encode_ok(42)) == 42


def test_error_roundtrip() -> None:
    code, message = proto.decode_error(proto.encode_error(proto.ERR_PARSE, "boom"))
    assert code == proto.ERR_PARSE
    assert message == "boom"


def test_resultset_roundtrip() -> None:
    result = ResultSet(
        columns=PAPERS,
        rows=(
            (1, 2019, "VLDB"),
            (2, 2020, None),
            (3, 2021, "ra\u00f1o \u00fc"),
        ),
    )
    decoded = proto.decode_resultset(proto.encode_resultset(result))
    assert decoded.columns == result.columns
    assert decoded.rows == result.rows


def test_resultset_all_types() -> None:
    schema = (
        ColumnDef("i", ColumnType.INT),
        ColumnDef("f", ColumnType.FLOAT),
        ColumnDef("v", ColumnType.VARCHAR, 20),
        ColumnDef("b", ColumnType.BOOL),
        ColumnDef("n", ColumnType.INT),
    )
    result = ResultSet(columns=schema, rows=((-5, 1.5, "x", True, None),))
    decoded = proto.decode_resultset(proto.encode_resultset(result))
    assert decoded.rows == ((-5, 1.5, "x", True, None),)
    assert decoded.columns[2].length == 20


def test_recv_frame_over_real_socket() -> None:
    left, right = socket.socketpair()
    try:
        right.sendall(proto.encode_frame(proto.OP_QUERY, proto.encode_query("SELECT 1")))
        opcode, payload = proto.recv_frame(left)
        assert opcode == proto.OP_QUERY
        assert proto.decode_query(payload) == "SELECT 1"
    finally:
        left.close()
        right.close()


def test_recv_frame_fragmented() -> None:
    left, right = socket.socketpair()
    try:
        frame = proto.encode_frame(proto.OP_PING)
        right.sendall(frame[:3])
        right.sendall(frame[3:])
        opcode, payload = proto.recv_frame(left)
        assert opcode == proto.OP_PING
        assert payload == b""
    finally:
        left.close()
        right.close()


def test_recv_frame_closed_socket_raises() -> None:
    left, right = socket.socketpair()
    right.close()
    with pytest.raises(ConnectionError):
        proto.recv_frame(left)
    left.close()


def test_error_code_mapping() -> None:
    assert proto.error_code(QueryParseError("x")) == proto.ERR_PARSE
    assert proto.error_code(QueryExecutionError("x")) == proto.ERR_EXECUTION
    assert proto.error_code(TransactionError("x")) == proto.ERR_TRANSACTION
    assert proto.error_code(DeadlockDetected("x")) == proto.ERR_DEADLOCK
    assert proto.error_code(ValueError("x")) == proto.ERR_GENERIC


def test_encode_unsupported_value_raises() -> None:
    with pytest.raises(ProtocolError):
        proto._encode_value(bytearray(), object())  # type: ignore[arg-type]
