"""Servidor ejecutable de la API de copyright sobre una base SoupDB ya preparada.

Arranque::

    uv run --group copyright-api python -m applications.copyright_detection.runtime
    # o, equivalente:
    uv run --group copyright-api uvicorn applications.copyright_detection.runtime:app

La configuración llega por variables ``COPYRIGHT_*``
(:mod:`applications.copyright_detection.settings`). El ciclo de vida usa el
``lifespan`` de FastAPI:

- **startup**: abre el ``Catalog`` con :func:`open_catalog` (factoría HNSW y
  perfil), comprueba con :func:`require_index` que el HNSW existe y está cargado,
  crea el :class:`CopyrightAnalyzer` y lo instala en ``app.state.analyzer``. Si
  algo falla, cierra lo que abrió y el servidor no arranca.
- **shutdown**: retira el analizador y cierra el ``Catalog`` una sola vez.

El runtime nunca entrena el codebook ni crea o reconstruye el índice: asume una
base preparada (por ejemplo con ``python -m applications.copyright_detection.demo
prepare``). Este módulo sí importa el pipeline multimedia; ``app.py`` sigue
siendo ligero.
"""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cache

from fastapi import FastAPI

from applications.copyright_detection import indexing
from applications.copyright_detection.analysis import CopyrightAnalyzer
from applications.copyright_detection.app import create_app
from applications.copyright_detection.indexing import CopyrightIndexConfig, CopyrightIndexError
from applications.copyright_detection.settings import (
    DB_PATH,
    VECTOR_STORAGE,
    RuntimeConfigError,
    RuntimeSettings,
)
from engine.common.catalog import Catalog

logger = logging.getLogger(__name__)


def index_config(settings: RuntimeSettings) -> CopyrightIndexConfig:
    return CopyrightIndexConfig(
        settings.dataset_root,
        settings.codebook_path,
        settings.descriptor_cache,
        settings.vector_storage,
    )


def open_analyzer(settings: RuntimeSettings) -> tuple[Catalog, CopyrightAnalyzer]:
    """Abre la base preparada y crea el analizador; no crea ni reconstruye nada.

    Devuelve el ``Catalog`` (que el caller debe cerrar) y el analizador. Si falla
    después de abrir el ``Catalog``, lo cierra antes de propagar el error.
    """
    # Catalog(...) crearía una base vacía en una ruta inexistente: se exige que exista.
    if not settings.db_path.is_dir():
        raise CopyrightIndexError(f"{DB_PATH} no es un directorio existente: {settings.db_path}")
    if not settings.vector_storage.is_dir():
        raise CopyrightIndexError(
            f"{VECTOR_STORAGE} no es un directorio existente: {settings.vector_storage}"
        )
    config = index_config(settings)
    catalog = indexing.open_catalog(settings.db_path, config)
    try:
        indexing.require_index(catalog, config.profile_id)
        analyzer = CopyrightAnalyzer(
            catalog,
            config,
            search_k=settings.search_k,
            staging_dir=settings.staging_dir,
        )
    except BaseException:
        catalog.close()
        raise
    return catalog, analyzer


def build_app(settings: RuntimeSettings) -> FastAPI:
    """App FastAPI cuyo ``lifespan`` abre la base al arrancar y la cierra al apagar."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        catalog, analyzer = open_analyzer(settings)
        logger.info(
            "copyright API lista: base %s, search_k=%d", settings.db_path, settings.search_k
        )
        app.state.analyzer = analyzer
        try:
            yield
        finally:
            app.state.analyzer = None
            catalog.close()

    return create_app(lifespan=lifespan, cors_origins=settings.cors_origins)


@cache
def _app_from_env() -> FastAPI:
    return build_app(RuntimeSettings.from_env())


def __getattr__(name: str) -> FastAPI:
    # ``uvicorn ...runtime:app``: la app se construye (leyendo el entorno) al pedirla,
    # no al importar el módulo; así importar ``runtime`` no exige configuración.
    if name == "app":
        return _app_from_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    try:
        settings = RuntimeSettings.from_env()
    except RuntimeConfigError as exc:
        sys.exit(f"configuración inválida: {exc}")
    import uvicorn  # solo al ejecutar el servidor

    uvicorn.run(build_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
