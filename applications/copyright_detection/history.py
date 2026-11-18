"""Historial de análisis persistido en SoupDB.

Dos tablas de aplicación, independientes de ``songs``/``audio_segments`` y
creadas en la misma base con SQL de SoupDB:

- ``analyses``: una fila por análisis (id, nombre del archivo, instante,
  duración de segmento y número de runs guardadas).
- ``analysis_matches``: una fila por run de evidencia (``CopyrightMatch``), con
  su posición ``match_order`` dentro del resultado.

Solo se guardan el resultado y la metadata: **nunca el audio subido**, que el
analizador borra al terminar. ``title`` y ``artist`` no se duplican: se leen de
``songs`` al reconstruir (el catálogo es de solo lectura en runtime).

Atomicidad: SoupDB no tiene ``UNIQUE`` ni transacciones durables entre tablas, así
que primero se insertan las runs y **al final** la fila de ``analyses``, que hace
de marca de commit. Las lecturas solo ven análisis con esa fila y comprueban que
el número de runs coincide con ``match_count``. Si una escritura falla a medias se
intenta borrar lo escrito; si quedara algo, son runs huérfanas que nunca se leen.

Durabilidad: el ``Catalog`` escribe las páginas a disco al desalojarlas del
buffer y al cerrarse (el motor aún no tiene WAL). Con un apagado normal del
servidor el historial persiste; un proceso matado de golpe puede perder los
análisis más recientes.

Concurrencia: el ``Catalog`` no es thread-safe; todas las operaciones usan el
``lock`` que se recibe, el mismo que el analizador usa para buscar.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from applications.copyright_detection.catalog import (
    CopyrightCatalog,
    IncompatibleSchemaError,
    IntegrityError,
    _sql_literal,
)
from applications.copyright_detection.schemas import (
    AnalysisResult,
    AnalysisSummary,
    CopyrightMatch,
    StoredAnalysis,
)
from engine.common.catalog import Catalog
from engine.common.schema import ColumnDef, ColumnType
from engine.query import ResultSet, execute_sql

ANALYSES = "analyses"
ANALYSIS_MATCHES = "analysis_matches"

ANALYSES_SCHEMA = (
    ColumnDef("analysis_id", ColumnType.TEXT),
    ColumnDef("filename", ColumnType.TEXT),
    ColumnDef("created_at", ColumnType.TEXT),
    ColumnDef("segment_seconds", ColumnType.FLOAT),
    ColumnDef("match_count", ColumnType.INT),
)

ANALYSIS_MATCHES_SCHEMA = (
    ColumnDef("analysis_id", ColumnType.TEXT),
    ColumnDef("match_order", ColumnType.INT),
    ColumnDef("song_id", ColumnType.INT),
    ColumnDef("query_start", ColumnType.FLOAT),
    ColumnDef("query_end", ColumnType.FLOAT),
    ColumnDef("source_start", ColumnType.FLOAT),
    ColumnDef("source_end", ColumnType.FLOAT),
    ColumnDef("score", ColumnType.FLOAT),
    ColumnDef("segment_count", ColumnType.INT),
    ColumnDef("min_score", ColumnType.FLOAT),
    ColumnDef("max_score", ColumnType.FLOAT),
    ColumnDef("alignment_offset_seconds", ColumnType.FLOAT),
    ColumnDef("is_best_run", ColumnType.BOOL),
)

#: Índices B+ para buscar por análisis. No son restricciones de unicidad.
_INDEXED_COLUMNS = {ANALYSES: ("analysis_id",), ANALYSIS_MATCHES: ("analysis_id",)}

#: ``analysis_id`` lo genera el analizador (``uuid4().hex``).
ANALYSIS_ID = re.compile(r"[0-9a-f]{32}")

#: ISO 8601 en UTC de ancho fijo: el orden de texto coincide con el cronológico.
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


class AnalysisHistoryError(Exception):
    """El historial no se pudo leer o escribir de forma consistente."""


def utc_now() -> str:
    return datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)


def is_analysis_id(value: str) -> bool:
    return ANALYSIS_ID.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class _StoredRow:
    analysis_id: str
    filename: str
    created_at: str
    segment_seconds: float
    match_count: int


class AnalysisHistory:
    """Guarda y reconstruye análisis en ``analyses``/``analysis_matches``.

    No abre ni cierra el ``Catalog``. Llamar a :meth:`initialize` antes de usarlo.
    """

    def __init__(
        self,
        catalog: Catalog,
        lock: "threading.RLock | None" = None,
        *,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self._catalog = catalog
        self._lock = lock if lock is not None else threading.RLock()
        self._clock = clock

    # --- bootstrap --------------------------------------------------------

    def initialize(self) -> None:
        """Crea las tablas e índices que falten y valida los existentes. Idempotente."""
        with self._lock:
            for table, schema in (
                (ANALYSES, ANALYSES_SCHEMA),
                (ANALYSIS_MATCHES, ANALYSIS_MATCHES_SCHEMA),
            ):
                if table in self._catalog.tables():
                    actual = self._catalog.schema(table)
                    if actual != schema:
                        raise IncompatibleSchemaError(
                            f"la tabla {table!r} existe con un schema incompatible"
                        )
                else:
                    columns = ", ".join(f"{c.name} {c.type_name.value}" for c in schema)
                    self._run(f"CREATE TABLE {table} ({columns})")
                for column in _INDEXED_COLUMNS[table]:
                    if not self._catalog.indexes_for(table, column):
                        self._run(f"CREATE INDEX {table}_{column} ON {table} ({column}) TYPE BTREE")

    # --- escritura ---------------------------------------------------------

    def save(self, result: AnalysisResult) -> StoredAnalysis:
        """Guarda ``result`` (runs primero, cabecera al final); lo devuelve con ``created_at``."""
        if not is_analysis_id(result.analysis_id):
            raise AnalysisHistoryError(f"analysis_id inválido: {result.analysis_id!r}")
        key = _sql_literal(result.analysis_id)
        with self._lock:
            if self._header(result.analysis_id) is not None:
                raise AnalysisHistoryError(f"el análisis {result.analysis_id} ya existe")
            # Bajo el lock: el orden de created_at coincide con el orden de escritura.
            created_at = self._clock()
            try:
                for order, match in enumerate(result.matches):
                    values = _match_values(key, order, match)
                    self._run(f"INSERT INTO {ANALYSIS_MATCHES} VALUES ({values})")
                header = ", ".join(
                    (
                        key,
                        _sql_literal(result.filename),
                        _sql_literal(created_at),
                        _sql_literal(float(result.segment_seconds)),
                        _sql_literal(len(result.matches)),
                    )
                )
                self._run(f"INSERT INTO {ANALYSES} VALUES ({header})")
            except BaseException:
                # Sin la cabecera las runs no se leen nunca; se limpian si se puede.
                try:
                    self._run(f"DELETE FROM {ANALYSIS_MATCHES} WHERE analysis_id = {key}")
                except Exception:  # noqa: BLE001 - la causa original es la relevante
                    pass
                raise
        return StoredAnalysis(**result.model_dump(), created_at=created_at)

    # --- lectura -----------------------------------------------------------

    def list_summaries(self) -> list[AnalysisSummary]:
        """Análisis guardados, del más reciente al más antiguo."""
        with self._lock:
            headers = [
                _StoredRow(*row)
                for row in self._run(
                    f"SELECT * FROM {ANALYSES} ORDER BY created_at DESC, analysis_id DESC"
                ).rows
            ]
            matches: dict[str, list[tuple]] = {}
            for row in self._run(
                f"SELECT analysis_id, song_id, score FROM {ANALYSIS_MATCHES}"
            ).rows:
                matches.setdefault(row[0], []).append(row)
        summaries = []
        for header in headers:
            rows = matches.get(header.analysis_id, [])
            if len(rows) != header.match_count:
                raise AnalysisHistoryError(
                    f"el análisis {header.analysis_id} tiene {len(rows)} runs; "
                    f"se esperaban {header.match_count}"
                )
            summaries.append(
                AnalysisSummary(
                    analysis_id=header.analysis_id,
                    filename=header.filename,
                    created_at=header.created_at,
                    segment_seconds=header.segment_seconds,
                    match_count=header.match_count,
                    song_count=len({row[1] for row in rows}),
                    best_score=max((row[2] for row in rows), default=None),
                )
            )
        return summaries

    def get(self, analysis_id: str) -> StoredAnalysis | None:
        """Reconstruye un análisis guardado, o ``None`` si no existe."""
        if not is_analysis_id(analysis_id):
            return None
        with self._lock:
            header = self._header(analysis_id)
            if header is None:
                return None
            rows = self._run(
                f"SELECT * FROM {ANALYSIS_MATCHES} "
                f"WHERE analysis_id = {_sql_literal(analysis_id)} ORDER BY match_order"
            ).rows
            if len(rows) != header.match_count:
                raise AnalysisHistoryError(
                    f"el análisis {analysis_id} tiene {len(rows)} runs; "
                    f"se esperaban {header.match_count}"
                )
            songs = CopyrightCatalog(self._catalog)
            matches = []
            for order, row in enumerate(rows):
                if row[1] != order:
                    raise AnalysisHistoryError(f"orden de runs inconsistente en {analysis_id}")
                song = songs.get_song(row[2])
                if song is None:
                    raise AnalysisHistoryError(
                        f"el análisis {analysis_id} referencia la canción inexistente {row[2]}"
                    )
                matches.append(_match_from_row(row, song.title, song.artist))
        return StoredAnalysis(
            analysis_id=header.analysis_id,
            filename=header.filename,
            created_at=header.created_at,
            segment_seconds=header.segment_seconds,
            matches=matches,
        )

    # --- internals ---------------------------------------------------------

    def _header(self, analysis_id: str) -> _StoredRow | None:
        rows = self._run(
            f"SELECT * FROM {ANALYSES} WHERE analysis_id = {_sql_literal(analysis_id)}"
        ).rows
        if not rows:
            return None
        if len(rows) > 1:
            raise IntegrityError(f"analysis_id {analysis_id} está duplicado en la base")
        return _StoredRow(*rows[0])

    def _run(self, sql: str) -> ResultSet:
        with self._lock:
            return execute_sql(sql, self._catalog)


def _match_values(key: str, order: int, match: CopyrightMatch) -> str:
    values = (
        _sql_literal(order),
        _sql_literal(match.song_id),
        _sql_literal(float(match.query_start)),
        _sql_literal(float(match.query_end)),
        _sql_literal(float(match.source_start)),
        _sql_literal(float(match.source_end)),
        _sql_literal(float(match.score)),
        _sql_literal(match.segment_count),
        _sql_literal(float(match.min_score)),
        _sql_literal(float(match.max_score)),
        _sql_literal(float(match.alignment_offset_seconds)),
        "TRUE" if match.is_best_run else "FALSE",
    )
    return ", ".join((key, *values))


def _match_from_row(row: tuple, title: str, artist: str) -> CopyrightMatch:
    (
        _analysis_id,
        _order,
        song_id,
        query_start,
        query_end,
        source_start,
        source_end,
        score,
        segment_count,
        min_score,
        max_score,
        offset,
        is_best_run,
    ) = row
    return CopyrightMatch(
        query_start=query_start,
        query_end=query_end,
        song_id=song_id,
        title=title,
        artist=artist,
        source_start=source_start,
        source_end=source_end,
        score=score,
        segment_count=segment_count,
        min_score=min_score,
        max_score=max_score,
        alignment_offset_seconds=offset,
        is_best_run=is_best_run,
    )
