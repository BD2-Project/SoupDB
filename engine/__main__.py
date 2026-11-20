"""Arranca el servidor TCP de SoupDB con configuración de entorno."""

import os
import signal
import sys
from pathlib import Path
from types import FrameType

from engine.common.catalog import Catalog
from engine.common.errors import UnsupportedOperation
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager
from server.documents import DocumentStore


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} debe ser un entero, no {raw!r}") from None


def main() -> None:
    data_dir = Path(os.environ.get("SOUPDB_DATA", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    document_root = os.environ.get("SOUPDB_DOCUMENT_ROOT") or None
    try:
        documents = DocumentStore(document_root) if document_root is not None else None
    except UnsupportedOperation as exc:
        raise SystemExit(f"SOUPDB_DOCUMENT_ROOT inválido: {exc}") from None

    catalog = Catalog(
        data_dir,
        page_size=_env_int("PAGE_SIZE", 4096),
        buffer_capacity=_env_int("BUFFER_POOL_SIZE", 1024),
    )
    handler = ConnectionHandler(
        catalog,
        TransactionManager(),
        host=os.environ.get("DRIVER_HOST", "0.0.0.0"),
        port=_env_int("DRIVER_PORT", 55432),
        max_connections=_env_int("MAX_CONNECTIONS", 100),
        lock_timeout_ms=_env_int("LOCK_TIMEOUT_MS", 5000),
        document_loader=documents,
    )
    handler.bind()

    def _shutdown(_signum: int, _frame: FrameType | None) -> None:
        # Docker manda SIGTERM al parar el contenedor: hay que cerrar los
        # archivos para no dejar la base a medio escribir.
        handler.shutdown()
        catalog.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    host = os.environ.get("DRIVER_HOST", "0.0.0.0")
    print(f"SoupDB escuchando en {host}:{handler.port}, datos en {data_dir}", flush=True)
    try:
        handler.serve_forever()
    finally:
        handler.shutdown()
        catalog.close()


if __name__ == "__main__":
    main()
