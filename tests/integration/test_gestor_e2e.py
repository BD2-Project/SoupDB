"""Suite de integración del gestor: conexiones TCP reales y transacciones.

Usa el Catalog real en disco (tmp_path) y el ConnectionHandler real sobre
puertos efímeros. Tests intensivos: corren por defecto en el CI; desactivar
con `SKIP_INTEGRATION=1` (ver conftest.py).
"""

import socket
import threading

import pytest

from engine.common.catalog import Catalog
from engine.query import execute_sql
from engine.transactions import protocol as proto
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager

pytestmark = pytest.mark.integration


@pytest.fixture
def database(tmp_path):
    catalog = Catalog(tmp_path / "db")
    execute_sql("CREATE TABLE accounts (id INT, balance INT)", catalog)
    execute_sql("INSERT INTO accounts VALUES (1, 100)", catalog)
    execute_sql("INSERT INTO accounts VALUES (2, 200)", catalog)
    yield catalog
    catalog.close()


@pytest.fixture
def server(database):
    handler = ConnectionHandler(
        database,
        TransactionManager(),
        host="127.0.0.1",
        port=0,
        max_connections=16,
        lock_timeout_ms=1000,
    )
    handler.bind()
    thread = threading.Thread(target=handler.serve_forever, daemon=True)
    thread.start()
    yield handler
    handler.shutdown()


def connect(server: ConnectionHandler) -> socket.socket:
    return socket.create_connection(("127.0.0.1", server.port), timeout=5)


def request(conn, opcode: int, payload: bytes = b"") -> tuple[int, bytes]:
    conn.sendall(proto.encode_frame(opcode, payload))
    return proto.recv_frame(conn)


def select_all(conn) -> tuple:
    opcode, payload = request(conn, proto.OP_QUERY, proto.encode_query("SELECT * FROM accounts"))
    assert opcode == proto.OP_RESULT
    return proto.decode_resultset(payload).rows


def test_full_lifecycle_over_tcp(server) -> None:
    conn = connect(server)
    try:
        assert request(conn, proto.OP_PING)[0] == proto.OP_PONG
        assert select_all(conn) == ((1, 100), (2, 200))

        request(conn, proto.OP_BEGIN)
        request(conn, proto.OP_QUERY, proto.encode_query("INSERT INTO accounts VALUES (90, 1000)"))
        assert request(conn, proto.OP_ROLLBACK)[0] == proto.OP_OK
        assert select_all(conn) == ((1, 100), (2, 200))

        request(conn, proto.OP_BEGIN)
        request(conn, proto.OP_QUERY, proto.encode_query("INSERT INTO accounts VALUES (91, 2000)"))
        assert request(conn, proto.OP_COMMIT)[0] == proto.OP_OK
        assert select_all(conn) == ((1, 100), (2, 200), (91, 2000))

        opcode, payload = request(conn, proto.OP_QUERY, proto.encode_query("BOGUS SQL"))
        assert opcode == proto.OP_ERROR
        code, _message = proto.decode_error(payload)
        assert code == proto.ERR_PARSE
    finally:
        conn.close()


def test_concurrent_clients_insert_and_commit(server) -> None:
    """Varios clientes insertan en transacciones simultáneamente.

    Cada cliente inserta filas únicas dentro de transacciones (BEGIN/COMMIT).
    El lock exclusivo de tabla serializa las escrituras sin formar ciclos
    (a diferencia del patrón read-modify-write con S→X), por lo que el test es
    determinista. Verifica el pool, las transacciones y el commit bajo carga.
    """
    threads = 8
    rounds = 5
    errors: list[Exception] = []
    guard = threading.Lock()

    def worker(index: int) -> None:
        conn = connect(server)
        try:
            for r in range(rounds):
                request(conn, proto.OP_BEGIN)
                account_id = 1000 + index * 100 + r
                opcode, payload = request(
                    conn,
                    proto.OP_QUERY,
                    proto.encode_query(
                        f"INSERT INTO accounts VALUES ({account_id}, {index * 10 + r})"
                    ),
                )
                assert opcode == proto.OP_RESULT, proto.decode_error(payload)
                request(conn, proto.OP_COMMIT)
        except Exception as exc:  # noqa: BLE001
            with guard:
                errors.append(exc)
        finally:
            conn.close()

    worker_threads = [threading.Thread(target=worker, args=(index,)) for index in range(threads)]
    for thread in worker_threads:
        thread.start()
    for thread in worker_threads:
        thread.join()

    assert not errors, errors
    conn = connect(server)
    try:
        ids = {row[0] for row in select_all(conn)}
    finally:
        conn.close()
    assert len(ids) == 2 + threads * rounds, len(ids)
    for index in range(threads):
        for r in range(rounds):
            assert 1000 + index * 100 + r in ids


def test_persistence_after_server_restart(database) -> None:
    def make_server():
        handler = ConnectionHandler(
            database,
            TransactionManager(),
            host="127.0.0.1",
            port=0,
            max_connections=4,
            lock_timeout_ms=1000,
        )
        handler.bind()
        thread = threading.Thread(target=handler.serve_forever, daemon=True)
        thread.start()
        return handler

    first = make_server()
    conn = connect(first)
    request(conn, proto.OP_BEGIN)
    request(conn, proto.OP_QUERY, proto.encode_query("INSERT INTO accounts VALUES (77, 777)"))
    request(conn, proto.OP_COMMIT)
    conn.close()
    first.shutdown()

    second = make_server()
    conn = connect(second)
    try:
        assert (77, 777) in select_all(conn)
    finally:
        conn.close()
    second.shutdown()
