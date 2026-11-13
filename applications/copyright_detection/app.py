"""Aplicación FastAPI del detector de copyright/remixes de audio.

La API REST es la frontera de la aplicación: el frontend solo consume
``/api/v1/...`` y nunca accede al ``Catalog`` ni a los archivos internos.

``create_app`` solo construye la aplicación HTTP. El analizador se lee de
``app.state.analyzer`` en cada petición: puede pasarse ya creado
(``create_app(analyzer)``) o instalarlo un ``lifespan``, que es lo que hace el
servidor real (:mod:`applications.copyright_detection.runtime`). Sin analizador,
``/health`` responde y ``/analyze`` devuelve 503.

Importar este módulo no abre bases, no lee variables de entorno, no entrena
codebooks ni carga el pipeline multimedia: ``analysis`` solo se importa para tipos.
La ``app`` global de este módulo no tiene analizador; el servidor ejecutable es
``python -m applications.copyright_detection.runtime``.
"""

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from applications.copyright_detection.config import API_PREFIX, SERVICE_NAME
from applications.copyright_detection.errors import AnalysisUnavailableError, InvalidAudioError
from applications.copyright_detection.schemas import (
    AnalysisResult,
    ErrorResponse,
    HealthResponse,
)

if TYPE_CHECKING:
    from applications.copyright_detection.analysis import CopyrightAnalyzer

logger = logging.getLogger(__name__)

ANALYZER_NOT_CONFIGURED = "El servicio de análisis no está configurado."
INTERNAL_ERROR = "Error interno del análisis."
#: Lo necesario para ``GET /health`` y ``POST /analyze`` multipart desde el navegador.
CORS_METHODS = ("GET", "POST")
CORS_HEADERS = ("Accept", "Content-Type")

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_app(
    analyzer: "CopyrightAnalyzer | None" = None,
    *,
    lifespan: Lifespan | None = None,
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    """Construye la API.

    - ``analyzer``: servicio ya configurado, o ``None`` (lo instala ``lifespan``
      en ``app.state.analyzer``, o ``/analyze`` responde 503).
    - ``lifespan``: ciclo de vida de FastAPI (abrir/cerrar recursos).
    - ``cors_origins``: orígenes del frontend permitidos. Vacío: sin CORS.
    """
    app = FastAPI(
        lifespan=lifespan,
        title=SERVICE_NAME,
        description=(
            "API de análisis de similitud acústica y temporal sobre SoupDB. Devuelve "
            "evidencia de similitud; no determina copyright."
        ),
        version="0.2.0",
    )
    app.state.analyzer = analyzer
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=list(CORS_METHODS),
            allow_headers=list(CORS_HEADERS),
            allow_credentials=False,
        )
    router = APIRouter(prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # Sin traceback para el cliente; queda en el log del servidor.
        logger.exception("error inesperado en la API de copyright", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": INTERNAL_ERROR})

    @router.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post(
        "/analyze",
        response_model=AnalysisResult,
        tags=["analysis"],
        summary="Analizar la similitud de un audio con el catálogo",
        responses={
            422: {
                "description": (
                    "Falta el campo `file`, o el audio está vacío, no se puede "
                    "decodificar o es demasiado corto para producir segmentos."
                ),
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "Análisis no configurado o índice/perfil no disponible.",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {
                "model": ErrorResponse,
                "description": "Error interno inesperado.",
            },
        },
    )
    def analyze(
        request: Request,
        file: UploadFile = File(description="Archivo de audio a analizar."),  # noqa: B008
    ) -> AnalysisResult:
        analyzer: CopyrightAnalyzer | None = request.app.state.analyzer
        if analyzer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, ANALYZER_NOT_CONFIGURED)
        try:
            return analyzer.analyze(file.file, file.filename)
        except InvalidAudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        except AnalysisUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    app.include_router(router)
    return app


app = create_app()
