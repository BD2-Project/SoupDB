"""Test the FETCH/DOCUMENT opcodes over the real TCP connection handler."""

import socket
import threading

import pytest

from engine.transactions import protocol as proto
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager
from server.documents import DocumentStore
from tests.fakes.fake_catalog import FakeCatalog


@pytest.fixture
def document_root(tmp_path):
    (tmp_path / "gatos").mkdir()
    (tmp_path / "gatos" / "001.jpg").write_bytes(b"\xff\xd8jpeg")
    return tmp_path


def start(document_root) -> ConnectionHandler:
    handler = ConnectionHandler(
        FakeCatalog(),
        TransactionManager(),
        host="127.0.0.1",
        port=0,
        max_connections=4,
        document_loader=DocumentStore(document_root),
    )
    handler.bind()
    threading.Thread(target=handler.serve_forever, daemon=True).start()
    return handler


def request(conn: socket.socket, opcode: int, payload: bytes = b"") -> tuple[int, bytes]:
    conn.sendall(proto.encode_frame(opcode, payload))
    return proto.recv_frame(conn)


def test_fetch_returns_bytes_and_media_type(document_root) -> None:
    handler = start(document_root)
    try:
        conn = socket.create_connection(("127.0.0.1", handler.port), timeout=5)
        try:
            opcode, payload = request(conn, proto.OP_FETCH, proto.encode_fetch("gatos/001.jpg"))
        finally:
            conn.close()
    finally:
        handler.shutdown()

    assert opcode == proto.OP_DOCUMENT
    assert proto.decode_document(payload) == ("image/jpeg", b"\xff\xd8jpeg")


def test_fetch_outside_root_is_record_not_found(document_root) -> None:
    handler = start(document_root)
    try:
        conn = socket.create_connection(("127.0.0.1", handler.port), timeout=5)
        try:
            opcode, payload = request(conn, proto.OP_FETCH, proto.encode_fetch("../secret.jpg"))
        finally:
            conn.close()
    finally:
        handler.shutdown()

    assert opcode == proto.OP_ERROR
    code, message = proto.decode_error(payload)
    assert code == proto.ERR_RECORD_NOT_FOUND
    assert "document reference" in message


def test_fetch_without_document_loader_is_unsupported() -> None:
    handler = ConnectionHandler(
        FakeCatalog(),
        TransactionManager(),
        host="127.0.0.1",
        port=0,
        max_connections=4,
    )
    handler.bind()
    threading.Thread(target=handler.serve_forever, daemon=True).start()
    try:
        conn = socket.create_connection(("127.0.0.1", handler.port), timeout=5)
        try:
            opcode, payload = request(conn, proto.OP_FETCH, proto.encode_fetch("gatos/001.jpg"))
        finally:
            conn.close()
    finally:
        handler.shutdown()

    assert opcode == proto.OP_ERROR
    code, _message = proto.decode_error(payload)
    assert code == proto.ERR_UNSUPPORTED
