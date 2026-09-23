"""Gestor TCP server: the connection pool that receives rsoup drivers.

The gestor owns the connection handling: a listener accepts up to
``MAX_CONNECTIONS`` clients; each connection runs on its own worker thread with
a :class:`TransactionalSession` that shares the server-wide
:class:`TransactionManager` and catalog, so concurrent clients are isolated by
strict 2PL locks. Connections beyond the limit wait in the accept backlog
(PgBouncer-style ``max_client_conn``).

Config via env (see ``docs/08-variables-entorno``): ``DRIVER_HOST``,
``DRIVER_PORT``, ``MAX_CONNECTIONS``, ``SOCKET_TIMEOUT`` and ``LOCK_TIMEOUT_MS``.
"""

import logging
import os
import socket
import threading
from typing import Any

from engine.common.errors import ProtocolError, SoupDBError
from engine.transactions import protocol as proto
from engine.transactions.session import TransactionalSession
from engine.transactions.transaction_manager import TransactionManager

_LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = os.getenv("DRIVER_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("DRIVER_PORT", "55432"))
DEFAULT_MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
DEFAULT_SOCKET_TIMEOUT = float(os.getenv("SOCKET_TIMEOUT", "30"))
DEFAULT_LOCK_TIMEOUT_MS = int(os.getenv("LOCK_TIMEOUT_MS", "3000"))


class ConnectionHandler:
    """Threaded TCP server exposing the engine behind the wire protocol."""

    def __init__(
        self,
        catalog: Any,
        transaction_manager: TransactionManager | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT,
        lock_timeout_ms: int = DEFAULT_LOCK_TIMEOUT_MS,
    ) -> None:
        self._catalog = catalog
        self._tm = transaction_manager or TransactionManager()
        self._host = host
        self._port = port
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._lock_timeout_ms = lock_timeout_ms
        self._listener: socket.socket | None = None
        self._slots = threading.BoundedSemaphore(max_connections)
        self._threads: list[threading.Thread] = []
        self._running = False

    @property
    def port(self) -> int:
        """The bound port (useful when ``port=0`` picks an ephemeral one)."""
        return self._port

    def bind(self) -> None:
        """Bind and listen on the configured host/port."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self._host, self._port))
        listener.listen(self._max_connections)
        self._listener = listener
        self._port = listener.getsockname()[1]

    def serve_forever(self) -> None:
        """Accept connections until :meth:`shutdown`."""
        if self._listener is None:
            self.bind()
        self._running = True
        while self._running:
            try:
                conn, addr = self._listener.accept()
            except OSError:
                if not self._running:
                    break
                raise
            conn.settimeout(self._socket_timeout)
            self._slots.acquire()
            thread = threading.Thread(target=self._handle, args=(conn, addr), daemon=True)
            self._threads.append(thread)
            thread.start()

    def shutdown(self) -> None:
        """Stop accepting connections and join worker threads."""
        self._running = False
        if self._listener is not None:
            self._listener.close()
        for thread in self._threads:
            thread.join(timeout=1)

    def _handle(self, conn: socket.socket, addr: Any) -> None:
        session = TransactionalSession(
            self._catalog,
            transaction_manager=self._tm,
            lock_timeout_ms=self._lock_timeout_ms,
        )
        try:
            while True:
                opcode, payload = proto.recv_frame(conn)
                conn.sendall(self._dispatch(opcode, payload, session))
        except ProtocolError as exc:
            try:
                conn.sendall(self._error(proto.ERR_GENERIC, str(exc)))
            except OSError:
                pass
        except (ConnectionError, OSError):
            pass
        finally:
            session.close()
            conn.close()
            self._slots.release()

    def _dispatch(
        self,
        opcode: int,
        payload: bytes,
        session: TransactionalSession,
    ) -> bytes:
        try:
            if opcode == proto.OP_PING:
                return proto.encode_frame(proto.OP_PONG)
            if opcode == proto.OP_BEGIN:
                session.begin()
                return proto.encode_frame(proto.OP_OK, proto.encode_ok(0))
            if opcode == proto.OP_COMMIT:
                session.commit()
                return proto.encode_frame(proto.OP_OK, proto.encode_ok(0))
            if opcode == proto.OP_ROLLBACK:
                session.rollback()
                return proto.encode_frame(proto.OP_OK, proto.encode_ok(0))
            if opcode == proto.OP_QUERY:
                sql = proto.decode_query(payload)
                result = session.execute(sql)
                return proto.encode_frame(proto.OP_RESULT, proto.encode_resultset(result))
            _LOGGER.warning("unknown opcode %s", opcode)
            return self._error(proto.ERR_GENERIC, f"unknown opcode {opcode}")
        except SoupDBError as exc:
            return self._error(proto.error_code(exc), str(exc))

    def _error(self, code: int, message: str) -> bytes:
        return proto.encode_frame(proto.OP_ERROR, proto.encode_error(code, message))
