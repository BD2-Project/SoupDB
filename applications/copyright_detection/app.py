"""Aplicación FastAPI del detector de copyright/remixes de audio.

Arranque local::

    uv run --group copyright-api uvicorn applications.copyright_detection.app:app --reload

``/analyze`` define el contrato pero responde ``501`` hasta que exista el motor
de detección: no devuelve coincidencias simuladas.
"""

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, status

from applications.copyright_detection.config import API_PREFIX, SERVICE_NAME
from applications.copyright_detection.schemas import (
    AnalysisResult,
    ErrorResponse,
    HealthResponse,
)

ANALYZE_NOT_IMPLEMENTED = "El motor de detección de copyright aún no está implementado."


def create_app() -> FastAPI:
    app = FastAPI(
        title=SERVICE_NAME,
        description="API de detección de copyright/remixes de audio sobre SoupDB.",
        version="0.1.0",
    )
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post(
        "/analyze",
        response_model=AnalysisResult,
        tags=["analysis"],
        summary="Analizar un audio contra el catálogo",
        responses={
            status.HTTP_501_NOT_IMPLEMENTED: {
                "model": ErrorResponse,
                "description": "El motor de detección aún no está implementado.",
            }
        },
    )
    def analyze(
        file: UploadFile = File(description="Archivo de audio a analizar."),  # noqa: B008
    ) -> AnalysisResult:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=ANALYZE_NOT_IMPLEMENTED
        )

    app.include_router(router)
    return app


app = create_app()
