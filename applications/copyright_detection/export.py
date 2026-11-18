"""Exportación de un análisis persistido a JSON o CSV.

Funciones puras (stdlib + pydantic): ``app.py`` las importa sin cargar el
pipeline multimedia.

- JSON: el :class:`~applications.copyright_detection.schemas.StoredAnalysis`
  completo, el mismo cuerpo que ``GET /api/v1/analyses/{analysis_id}``.
- CSV: una fila por run de evidencia, en el orden del resultado. Sin runs solo
  hay cabecera. ``duration`` es ``query_end - query_start`` (s). Los campos de
  texto que empiezan por ``= + - @`` o un control se prefijan con ``'`` para que
  una hoja de cálculo no los ejecute como fórmula (el nombre del archivo lo elige
  el usuario).

No hay umbrales, veredictos ni probabilidades: ``score`` es una similitud.
"""

import csv
import io
import re

from applications.copyright_detection.schemas import StoredAnalysis

EXPORT_FORMATS = ("json", "csv")
CSV_COLUMNS = (
    "analysis_id",
    "created_at",
    "filename",
    "match_order",
    "song_id",
    "title",
    "artist",
    "query_start",
    "query_end",
    "source_start",
    "source_end",
    "duration",
    "score",
    "segment_count",
    "min_score",
    "max_score",
    "alignment_offset_seconds",
    "is_best_run",
)
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_text(value: str) -> str:
    return "'" + value if value.startswith(_FORMULA_START) else value


def to_json(analysis: StoredAnalysis) -> str:
    return analysis.model_dump_json(indent=2)


def to_csv(analysis: StoredAnalysis) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for order, match in enumerate(analysis.matches):
        writer.writerow(
            (
                analysis.analysis_id,
                analysis.created_at,
                _safe_text(analysis.filename),
                order,
                match.song_id,
                _safe_text(match.title),
                _safe_text(match.artist),
                repr(match.query_start),
                repr(match.query_end),
                repr(match.source_start),
                repr(match.source_end),
                repr(round(match.query_end - match.query_start, 6)),
                repr(match.score),
                match.segment_count,
                repr(match.min_score),
                repr(match.max_score),
                repr(match.alignment_offset_seconds),
                "true" if match.is_best_run else "false",
            )
        )
    return buffer.getvalue()


def export_filename(analysis_id: str, fmt: str) -> str:
    """Nombre de descarga generado en servidor (el id ya está validado; se filtra igual)."""
    return f"analysis-{_UNSAFE_FILENAME_CHARS.sub('', analysis_id)}.{fmt}"
