"""Contrato HTTP del API de copyright.

Cada ``CopyrightMatch`` es una run de evidencia temporal: un tramo continuo del
audio analizado alineado con un tramo de una canción del catálogo. ``score`` es la
**media de los scores de similitud** de los segmentos de esa run, no una
probabilidad de plagio, copyright ni infracción. No existe un umbral de copyright
validado experimentalmente, así que el contrato no expone ninguno.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from applications.copyright_detection.config import SERVICE_NAME

#: Holgura para comparar la media con el mínimo/máximo: ``statistics.fmean`` puede
#: quedar un ulp por encima del máximo (``fmean([0.1] * 3) > 0.1``).
_SCORE_TOLERANCE = 1e-9


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: Literal["ok"] = "ok"
    service: str = SERVICE_NAME


class CopyrightMatch(BaseModel):
    """Run de evidencia: intervalo del audio analizado alineado con una canción del catálogo."""

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
            "Media de los scores de similitud de los segmentos de la run "
            "(euclidean: 1 / (1 + distancia); mayor = más parecido). "
            "No es una probabilidad de plagio, copyright ni infracción."
        ),
    )
    segment_count: int = Field(
        ge=1, strict=True, description="Segmentos consecutivos de 5 s que forman la run."
    )
    min_score: float = Field(ge=0, le=1, description="Menor score de similitud de la run.")
    max_score: float = Field(ge=0, le=1, description="Mayor score de similitud de la run.")
    alignment_offset_seconds: float = Field(
        description="source_start - query_start: desplazamiento del tramo en la canción (s)."
    )
    is_best_run: bool = Field(
        strict=True, description="Si es la run con más evidencia temporal de esa canción."
    )

    @model_validator(mode="after")
    def _intervals_are_ordered(self) -> Self:
        if self.query_end <= self.query_start:
            raise ValueError("query_end debe ser > query_start (duración positiva)")
        if self.source_end <= self.source_start:
            raise ValueError("source_end debe ser > source_start (duración positiva)")
        if self.min_score > self.max_score:
            raise ValueError("min_score debe ser <= max_score")
        if not (
            self.min_score - _SCORE_TOLERANCE <= self.score <= self.max_score + _SCORE_TOLERANCE
        ):
            raise ValueError("score debe estar entre min_score y max_score")
        return self


class AnalysisResult(BaseModel):
    """Resultado de analizar un audio contra el catálogo."""

    analysis_id: str = Field(description="Identificador del análisis.")
    filename: str = Field(description="Nombre del archivo subido (solo el nombre, sin ruta).")
    segment_seconds: float = Field(gt=0, description="Duración de cada segmento analizado (s).")
    matches: list[CopyrightMatch] = Field(
        default_factory=list,
        description=(
            "Runs de evidencia temporal, agrupadas por canción (más coherente primero) y "
            "en orden temporal dentro de cada canción. Vacía si no hay evidencia."
        ),
    )


class ErrorResponse(BaseModel):
    """Error devuelto por el API (formato estándar de FastAPI)."""

    detail: str
