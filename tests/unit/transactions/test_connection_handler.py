"""Tests for the gestor TCP connection handler (connection pool)."""

import socket
import threading

import pytest

from engine.common.schema import ColumnDef, ColumnType
from engine.transactions import protocol as proto
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager
from tests.fakes.fake_catalog import FakeCatalog

ACCOUNTS = (ColumnDef("id", ColumnType.INT), ColumnDef("balance", ColumnType.INT))


def make_catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("accounts", ACCOUNTS)
    catalog.insert("accounts", (1, 100))
    catalog.insert("accounts", (2, 200))
    return catalog


@pytest.fixture
def server():
    catalog = make_catalog()
    handler = ConnectionHandler(
        catalog,
        TransactionManager(),
        host="127.0.0.1",
        port=0,
        max_connections=4,
        lock_timeout_ms=100,
    )
    handler.bind()
    thread = threading.Thread(target=handler.serve_forever, daemon=True)
    thread.start()
    yield handler
    handler.shutdown()


def connect(server: ConnectionHandler) -> socket.socket:
    conn = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    return conn


def request(conn: socket.socket, opcode: int, payload: bytes = b"") -> tuple[int, bytes]:
    conn.sendall(proto.encode_frame(opcode, payload))
    return proto.recv_frame(conn)


def test_ping_returns_pong(server) -> None:
    conn = connect(server)
    try:
        opcode, _payload = request(conn, proto.OP_PING)
        assert opcode == proto.OP_PONG
    finally:
        conn.close()


def test_query_returns_resultset(server) -> None:
    conn = connect(server)
    try:
        opcode, payload = request(
            conn, proto.OP_QUERY, proto.encode_query("SELECT * FROM accounts")
        )
        assert opcode == proto.OP_RESULT
        result = proto.decode_resultset(payload)
        assert result.rows == ((1, 100), (2, 200))
    finally:
        conn.close()


def test_insert_within_transaction_and_rollback(server) -> None:
    conn = connect(server)
    try:
        assert request(conn, proto.OP_BEGIN)[0] == proto.OP_OK
        assert (
            request(
                conn, proto.OP_QUERY, proto.encode_query("INSERT INTO accounts VALUES (3, 300)")
            )[0]
            == proto.OP_RESULT
        )
        assert request(conn, proto.OP_ROLLBACK)[0] == proto.OP_OK
        _opcode, payload = request(
            conn, proto.OP_QUERY, proto.encode_query("SELECT * FROM accounts")
        )
        assert proto.decode_resultset(payload).rows == ((1, 100), (2, 200))
    finally:
        conn.close()


def test_insert_within_transaction_and_commit(server) -> None:
    conn = connect(server)
    try:
        assert request(conn, proto.OP_BEGIN)[0] == proto.OP_OK
        assert (
            request(
                conn, proto.OP_QUERY, proto.encode_query("INSERT INTO accounts VALUES (3, 300)")
            )[0]
            == proto.OP_RESULT
        )
        assert request(conn, proto.OP_COMMIT)[0] == proto.OP_OK
        _opcode, payload = request(
            conn, proto.OP_QUERY, proto.encode_query("SELECT * FROM accounts")
        )
        assert proto.decode_resultset(payload).rows == ((1, 100), (2, 200), (3, 300))
    finally:
        conn.close()


def test_query_error_returns_error_frame(server) -> None:
    conn = connect(server)
    try:
        opcode, payload = request(conn, proto.OP_QUERY, proto.encode_query("BOGUS SQL"))
        assert opcode == proto.OP_ERROR
        code, message = proto.decode_error(payload)
        assert code == proto.ERR_PARSE
        assert message
    finally:
        conn.close()


def test_bad_magic_raises_protocol_error(server) -> None:
    conn = connect(server)
    try:
        conn.sendall(b"XX\x01\x06\x00\x00\x00\x00")
        opcode, payload = proto.recv_frame(conn)
        assert opcode == proto.OP_ERROR
        code, _message = proto.decode_error(payload)
        assert code == proto.ERR_GENERIC
    finally:
        conn.close()


def test_concurrent_clients_share_concurrency_control(server) -> None:
    conn_a = connect(server)
    conn_b = connect(server)
    try:
        assert request(conn_a, proto.OP_BEGIN)[0] == proto.OP_OK
        assert (
            request(
                conn_a, proto.OP_QUERY, proto.encode_query("INSERT INTO accounts VALUES (9, 999)")
            )[0]
            == proto.OP_RESULT
        )
        opcode, payload = request(
            conn_b, proto.OP_QUERY, proto.encode_query("SELECT * FROM accounts")
        )
        assert opcode == proto.OP_ERROR
        code, _message = proto.decode_error(payload)
        assert code in (proto.ERR_TRANSACTION, proto.ERR_LOCK_TIMEOUT, proto.ERR_DEADLOCK)
        assert request(conn_a, proto.OP_ROLLBACK)[0] == proto.OP_OK
        opcode, payload = request(
            conn_b, proto.OP_QUERY, proto.encode_query("SELECT * FROM accounts")
        )
        assert opcode == proto.OP_RESULT
    finally:
        conn_a.close()
        conn_b.close()
