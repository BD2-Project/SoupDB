"""Punto de entrada del gestor: ``python -m engine``.

Levanta el servidor TCP que habla el protocolo binario con el driver y con
SoupChef. Sin esto el motor solo se puede usar escribiendo Python a mano, y el
contenedor no sirve nada en su puerto.

Toda la configuración llega por variables de entorno, que es lo que espera
``docker-compose.yml``:

====================  ==========================  ======================
Variable              Valor por defecto           Qué controla
====================  ==========================  ======================
``SOUPDB_DATA``       ``data``                    Directorio de la base
``DRIVER_HOST``       ``0.0.0.0``                 Interfaz de escucha
``DRIVER_PORT``       ``55432``                   Puerto
``PAGE_SIZE``         ``4096``                    Tamaño de página
``BUFFER_POOL_SIZE``  ``1024``                    Páginas en el buffer pool
``MAX_CONNECTIONS``   ``100``                     Conexiones simultáneas
``LOCK_TIMEOUT_MS``   ``5000``                    Espera máxima por un lock
====================  ==========================  ======================

Escucha en ``0.0.0.0`` y no en ``127.0.0.1`` porque dentro de un contenedor una
escucha en loopback no es alcanzable desde el host.
"""

import os
import signal
import sys
from pathlib import Path
from types import FrameType

from engine.common.catalog import Catalog
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager


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
