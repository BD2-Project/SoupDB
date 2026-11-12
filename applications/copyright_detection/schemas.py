"""Contrato HTTP del API de copyright.

``score`` es un score de **similitud**, no una probabilidad de plagio, copyright
ni infracción. Todavía no existe un umbral de copyright validado
experimentalmente, así que el contrato no expone ninguno.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from applications.copyright_detection.config import SERVICE_NAME


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: Literal["ok"] = "ok"
    service: str = SERVICE_NAME


class CopyrightMatch(BaseModel):
    """Coincidencia entre un intervalo del audio consultado y una canción del catálogo."""

    query_start: float = Field(ge=0, description="Inicio del intervalo en el audio consultado (s).")
    query_end: float = Field(ge=0, description="Fin del intervalo en el audio consultado (s).")
    song_id: int = Field(description="Identificador de la canción del catálogo.")
    title: str = Field(description="Título de la canción del catálogo.")
    artist: str = Field(description="Artista de la canción del catálogo.")
    source_start: float = Field(ge=0, description="Inicio del intervalo en la canción (s).")
    source_end: float = Field(ge=0, description="Fin del intervalo en la canción (s).")
    score: float = Field(
        ge=0,
        le=1,
        description=(
            "Score de similitud entre ambos intervalos (mayor = más parecido). "
            "No es una probabilidad de plagio, copyright ni infracción."
        ),
    )

    @model_validator(mode="after")
    def _intervals_are_ordered(self) -> Self:
        if self.query_end <= self.query_start:
            raise ValueError("query_end debe ser > query_start (duración positiva)")
        if self.source_end <= self.source_start:
            raise ValueError("source_end debe ser > source_start (duración positiva)")
        return self


class AnalysisResult(BaseModel):
    """Resultado de analizar un audio contra el catálogo."""

    analysis_id: str = Field(description="Identificador del análisis.")
    filename: str = Field(description="Nombre del archivo subido.")
    segment_seconds: float = Field(gt=0, description="Duración de cada segmento analizado (s).")
    matches: list[CopyrightMatch] = Field(
        default_factory=list, description="Coincidencias encontradas."
    )


class ErrorResponse(BaseModel):
    """Error devuelto por el API (formato estándar de FastAPI)."""

    detail: str
