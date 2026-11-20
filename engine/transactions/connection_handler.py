"""Gestor TCP server: the connection pool that receives rsoup drivers.

The gestor owns the connection handling: a listener accepts up to
``MAX_CONNECTIONS`` clients; each connection runs on its own worker thread with
a :class:`TransactionalSession` that shares the server-wide
:class:`TransactionManager` and catalog, so concurrent clients are isolated by
strict 2PL locks. Connections beyond the limit wait in the accept backlog
(PgBouncer-style ``max_client_conn``).

Config via env (see ``docs/08-variables-entorno``): ``DRIVER_HOST``,
``DRIVER_PORT``, ``MAX_CONNECTIONS``, ``SOCKET_TIMEOUT``,
``LOCK_TIMEOUT_MS`` and ``SOUPDB_DOCUMENT_ROOT`` (raíz de los documentos que
sirve ``OP_FETCH``; sin ella, ``FETCH`` responde ``UNSUPPORTED``).
"""

import logging
import os
import socket
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from engine.common.errors import ProtocolError, SoupDBError
from engine.query.params import bind_parameters
from engine.query.parser import parse
from engine.transactions import protocol as proto
from engine.transactions.session import TransactionalSession
from engine.transactions.transaction_manager import TransactionManager

_LOGGER = logging.getLogger(__name__)


class DocumentHandle(Protocol):
    """Documento resuelto que el servidor lee en trozos (sin cargarlo entero)."""

    media_type: str
    size: int

    def read_chunks(self, chunk_size: int = ...) -> Iterable[bytes]: ...


class DocumentLoader(Protocol):
    """Resuelve una referencia relativa a un documento bajo la raíz global."""

    def resolve(self, reference: str) -> DocumentHandle: ...


@dataclass(frozen=True, slots=True)
class DocumentResponse:
    """Respuesta ``DOCUMENT`` que se envía en streaming por el socket."""

    payload_length: int
    header: bytes
    chunks: Iterable[bytes]


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
        document_loader: DocumentLoader | None = None,
    ) -> None:
        self._catalog = catalog
        self._tm = transaction_manager or TransactionManager()
        self._host = host
        self._port = port
        self._max_connections = max_connections
        self._socket_timeout = socket_timeout
        self._lock_timeout_ms = lock_timeout_ms
        # El motor no toca el filesystem de documentos: el loader se inyecta.
        self._documents = document_loader
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
            prepared: dict[str, Any] = {}
            while True:
                version, opcode, payload = proto.recv_frame_versioned(conn)
                if opcode == proto.OP_PREPARE:
                    conn.sendall(self._prepare(payload, prepared, version))
                elif opcode == proto.OP_EXECUTE:
                    conn.sendall(self._execute_prepared(payload, prepared, session, version))
                elif opcode == proto.OP_DEALLOCATE:
                    conn.sendall(self._deallocate(payload, prepared, version))
                else:
                    response = self._dispatch(opcode, payload, session, version)
                    if isinstance(response, DocumentResponse):
                        self._send_document(conn, response, version)
                    else:
                        conn.sendall(response)
        except ProtocolError as exc:
            try:
                conn.sendall(self._error(proto.ERR_GENERIC, str(exc), proto.VERSION))
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
        version: int = proto.VERSION,
    ) -> bytes | DocumentResponse:
        try:
            if opcode == proto.OP_PING:
                return proto.encode_frame(proto.OP_PONG, version=version)
            if opcode == proto.OP_BEGIN:
                session.begin()
                return proto.encode_frame(proto.OP_OK, proto.encode_ok(0), version=version)
            if opcode == proto.OP_COMMIT:
                session.commit()
                return proto.encode_frame(proto.OP_OK, proto.encode_ok(0), version=version)
            if opcode == proto.OP_ROLLBACK:
                session.rollback()
                return proto.encode_frame(proto.OP_OK, proto.encode_ok(0), version=version)
            if opcode == proto.OP_QUERY:
                sql = proto.decode_query(payload)
                result = session.execute(sql)
                return self._result_frame(result, version)
            if opcode == proto.OP_FETCH:
                return self._document_response(payload, version)
            _LOGGER.warning("unknown opcode %s", opcode)
            return self._error(proto.ERR_GENERIC, f"unknown opcode {opcode}", version)
        except SoupDBError as exc:
            return self._error(proto.error_code(exc), str(exc), version)

    def _prepare(self, payload: bytes, prepared: dict[str, Any], version: int) -> bytes:
        try:
            name, sql = proto.decode_prepare(payload)
            prepared[name] = parse(sql)
            return proto.encode_frame(proto.OP_OK, proto.encode_ok(0), version=version)
        except SoupDBError as exc:
            return self._error(proto.error_code(exc), str(exc), version)

    def _execute_prepared(
        self,
        payload: bytes,
        prepared: dict[str, Any],
        session: TransactionalSession,
        version: int,
    ) -> bytes:
        try:
            name, params = proto.decode_execute(payload)
            statement = prepared.get(name)
            if statement is None:
                return self._error(
                    proto.ERR_GENERIC, f"unknown prepared statement {name!r}", version
                )
            result = session.execute_statement(bind_parameters(statement, params))
            return self._result_frame(result, version)
        except SoupDBError as exc:
            return self._error(proto.error_code(exc), str(exc), version)

    def _deallocate(self, payload: bytes, prepared: dict[str, Any], version: int) -> bytes:
        try:
            name = proto.decode_deallocate(payload)
            prepared.pop(name, None)
            return proto.encode_frame(proto.OP_OK, proto.encode_ok(0), version=version)
        except SoupDBError as exc:
            return self._error(proto.error_code(exc), str(exc), version)

    def _result_frame(self, result: Any, version: int) -> bytes:
        # Un SELECT devuelve RESULT; DML y DDL devuelven OK con las filas
        # afectadas, como fija docs/protocolo.md y espera el driver rsoup.
        if result.columns:
            return proto.encode_frame(
                proto.OP_RESULT,
                proto.encode_resultset(result, version=version),
                version=version,
            )
        return proto.encode_frame(proto.OP_OK, proto.encode_ok(result.affected), version=version)

    def _document_response(self, payload: bytes, version: int) -> bytes | DocumentResponse:
        """Prepara el ``DOCUMENT`` de un documento bajo ``SOUPDB_DOCUMENT_ROOT``.

        El cliente solo manda la referencia relativa; la raíz la conoce el gestor.
        Así el mismo gestor sirve en local o en un contenedor con el volumen
        montado, sin que el cliente reconstruya rutas. El archivo se lee en
        trozos; el motor no carga el documento completo en memoria.
        """
        if self._documents is None:
            return self._error(proto.ERR_UNSUPPORTED, "document root not configured", version)
        try:
            reference = proto.decode_fetch(payload)
            document = self._documents.resolve(reference)
            header = proto.encode_document_header(document.media_type, document.size)
            return DocumentResponse(
                payload_length=len(header) + document.size,
                header=header,
                chunks=document.read_chunks(),
            )
        except OSError as exc:
            return self._error(proto.ERR_EXECUTION, f"cannot read document: {exc}", version)
        except SoupDBError as exc:
            return self._error(proto.error_code(exc), str(exc), version)

    def _send_document(self, conn: socket.socket, response: DocumentResponse, version: int) -> None:
        conn.sendall(
            proto.encode_frame_header(proto.OP_DOCUMENT, response.payload_length, version=version)
        )
        conn.sendall(response.header)
        for chunk in response.chunks:
            if chunk:
                conn.sendall(chunk)

    def _error(self, code: int, message: str, version: int = proto.VERSION) -> bytes:
        return proto.encode_frame(
            proto.OP_ERROR, proto.encode_error(code, message), version=version
        )
