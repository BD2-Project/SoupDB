"""Tests for the binary row codec."""

import struct

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType

SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("score", ColumnType.FLOAT),
    ColumnDef("titulo", ColumnType.VARCHAR, length=32),
    ColumnDef("cuerpo", ColumnType.TEXT),
    ColumnDef("activo", ColumnType.BOOL),
)
ROW = (7, 0.9, "RAG", "resumen del paper", True)


def test_encode_decode_roundtrip() -> None:
    data = encode_row(ROW, SCHEMA)
    assert isinstance(data, bytes)
    assert decode_row(data, SCHEMA) == ROW


def test_encode_empty_schema() -> None:
    data = encode_row((), ())
    assert data == b""
    assert decode_row(data, ()) == ()


def test_encode_int_little_endian() -> None:
    data = encode_row((1,), (ColumnDef("id", ColumnType.INT),))
    assert data == (1).to_bytes(4, "little")


def test_decode_int() -> None:
    data = (2020).to_bytes(4, "little")
    assert decode_row(data, (ColumnDef("id", ColumnType.INT),)) == (2020,)


def test_encode_float() -> None:
    import struct

    data = encode_row((3.5,), (ColumnDef("score", ColumnType.FLOAT),))
    assert data == struct.pack("<d", 3.5)


def test_decode_float() -> None:
    import struct

    data = struct.pack("<d", -1.25)
    assert decode_row(data, (ColumnDef("score", ColumnType.FLOAT),)) == (-1.25,)


def test_encode_bool() -> None:
    assert encode_row((True,), (ColumnDef("ok", ColumnType.BOOL),)) == b"\x01"
    assert encode_row((False,), (ColumnDef("ok", ColumnType.BOOL),)) == b"\x00"


def test_decode_bool() -> None:
    schema = (ColumnDef("ok", ColumnType.BOOL),)
    assert decode_row(b"\x01", schema) == (True,)
    assert decode_row(b"\x00", schema) == (False,)


def test_encode_varchar_with_length_prefix() -> None:
    import struct

    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=32),)
    data = encode_row(("RAG",), schema)
    (length,) = struct.unpack("<I", data[:4])
    assert length == 3
    assert data[4:] == b"RAG"


def test_encode_varchar_utf8_length() -> None:
    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=32),)
    data = encode_row(("ñandú",), schema)
    assert decode_row(data, schema) == ("ñandú",)


def test_encode_varchar_over_length_raises() -> None:
    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=5),)
    with pytest.raises(QueryExecutionError):
        encode_row(("demasiado largo",), schema)


def test_decode_text() -> None:
    schema = (ColumnDef("cuerpo", ColumnType.TEXT),)
    data = encode_row(("cuerpo de muchas palabras",), schema)
    assert decode_row(data, schema) == ("cuerpo de muchas palabras",)


def test_encode_wrong_int_type_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    with pytest.raises(QueryExecutionError):
        encode_row(("1",), schema)


def test_encode_bool_as_int_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    with pytest.raises(QueryExecutionError):
        encode_row((True,), schema)


def test_encode_int_as_float_raises() -> None:
    schema = (ColumnDef("score", ColumnType.FLOAT),)
    with pytest.raises(QueryExecutionError):
        encode_row((1,), schema)


def test_decode_truncated_data_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    with pytest.raises(QueryExecutionError):
        decode_row(b"\x01\x02", schema)


def test_decode_trailing_bytes_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    data = encode_row((1,), schema) + b"\x00"
    with pytest.raises(QueryExecutionError):
        decode_row(data, schema)


def test_encode_wrong_row_length_raises() -> None:
    with pytest.raises(QueryExecutionError):
        encode_row((1,), SCHEMA)


def test_decode_with_invalid_bool_byte_raises() -> None:
    schema = (ColumnDef("activo", ColumnType.BOOL),)
    with pytest.raises(QueryExecutionError):
        decode_row(b"\x02", schema)


def test_decode_invalid_utf8_text_raises() -> None:
    raw = struct.pack("<I", 2) + b"\xff\xfe"
    with pytest.raises(QueryExecutionError, match="UTF-8"):
        decode_row(raw, (ColumnDef("txt", ColumnType.TEXT),))


def test_record_holds_encoded_bytes() -> None:
    data = encode_row(ROW, SCHEMA)
    record = Record(data=data)
    assert decode_row(record.data, SCHEMA) == ROW
