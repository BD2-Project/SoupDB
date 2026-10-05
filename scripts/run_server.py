"""Arranca el servidor TCP del gestor (pool de conexiones).

Lee la configuracion desde variables de entorno (ver docs/08):
DRIVER_HOST, DRIVER_PORT, MAX_CONNECTIONS, SOCKET_TIMEOUT, LOCK_TIMEOUT_MS
y SOUP_DB_PATH (ruta de la base de datos, default ./data).

Uso: uv run python scripts/run_server.py
"""

import logging
import os

from engine.common.catalog import Catalog
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager

_DEFAULT_DB_PATH = os.getenv("SOUP_DB_PATH", "./data")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    catalog = Catalog(_DEFAULT_DB_PATH)
    handler = ConnectionHandler(catalog, TransactionManager())
    logging.info(
        "gestor escuchando en %s:%s (max %s)",
        handler._host,
        handler._port,
        handler._max_connections,
    )
    handler.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
