"""Catálogo relacional de la aplicación de copyright sobre el motor real de SoupDB.

Dos tablas de aplicación, creadas con SQL de SoupDB sobre un
:class:`engine.common.catalog.Catalog` persistente:

- ``songs``: canciones originales del catálogo.
- ``audio_segments``: segmentos indexables de cada canción. ``audio_ref`` sigue
  la convención de la Parte 4 (``soupdb_vector.multimedia.audio_features.segment_id``):
  ``<audio_path>#segment=NNNNNN``. Es la referencia que usará después el
  pipeline multimedia; aquí todavía no hay índice HNSW ni vectores.

SoupDB no tiene ``UNIQUE``, ``FOREIGN KEY``, ``IF NOT EXISTS`` ni ``UPDATE``
ejecutable, y su ``PRIMARY KEY`` crea un índice B+ pero no impide duplicados.
Por eso esta capa comprueba las invariantes antes de cada ``INSERT`` y hace el
bootstrap idempotente consultando el catálogo.

Limitación conocida: el ``RLock`` solo serializa los accesos de esta instancia
dentro de un proceso. No sustituye restricciones del motor ni coordina varios
procesos, y registrar una canción y sus segmentos no es atómico. Se resolverá al
integrar la ingesta con transacciones.

Las sentencias se ejecutan con la API pública ``engine.query.execute_sql``, que
no acepta parámetros: los valores se serializan con :func:`_sql_literal`, que
solo admite ``str``, ``int`` y ``float`` exactos.
"""

import math
import threading
from dataclasses import astuple, dataclass
from decimal import Decimal
from pathlib import PurePosixPath, PureWindowsPath

from engine.common.catalog import Catalog
from engine.common.schema import ColumnDef, ColumnType
from engine.query import ResultSet, execute_sql

SONGS = "songs"
AUDIO_SEGMENTS = "audio_segments"

SONGS_SCHEMA = (
    ColumnDef("song_id", ColumnType.INT),
    ColumnDef("title", ColumnType.TEXT),
    ColumnDef("artist", ColumnType.TEXT),
    ColumnDef("audio_path", ColumnType.TEXT),
    ColumnDef("duration_seconds", ColumnType.FLOAT),
)

AUDIO_SEGMENTS_SCHEMA = (
    ColumnDef("segment_id", ColumnType.INT),
    ColumnDef("song_id", ColumnType.INT),
    ColumnDef("segment_index", ColumnType.INT),
    ColumnDef("start_seconds", ColumnType.FLOAT),
    ColumnDef("end_seconds", ColumnType.FLOAT),
    ColumnDef("audio_ref", ColumnType.TEXT),
)

#: Índices B+ por tabla: aceleran la búsqueda por id y por canción. No son
#: restricciones de unicidad (SoupDB no las tiene).
_INDEXED_COLUMNS = {
    SONGS: ("song_id",),
    AUDIO_SEGMENTS: ("segment_id", "song_id"),
}

_SEGMENT_MARKER = "#segment="


class CopyrightCatalogError(Exception):
    """Error de la capa de catálogo de copyright."""


class IncompatibleSchemaError(CopyrightCatalogError):
    """Una tabla de la aplicación ya existe con un schema distinto del esperado."""


class IntegrityError(CopyrightCatalogError):
    """La operación rompería una invariante del catálogo (duplicado, padre inexistente)."""


def segment_audio_ref(audio_path: str, segment_index: int) -> str:
    """Referencia canónica de un segmento: ``<audio_path>#segment=NNNNNN``.

    Misma forma que ``soupdb_vector.multimedia.audio_features.segment_id`` (un
    test lo verifica). No se importa de ahí para no cargar librosa en la API.
    """
    return f"{audio_path}{_SEGMENT_MARKER}{segment_index:06d}"


def _require_positive_id(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} debe ser un entero positivo, no {value!r}")


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} debe ser un texto no vacío")


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito, no {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class Song:
    song_id: int
    title: str
    artist: str
    audio_path: str
    duration_seconds: float

    def __post_init__(self) -> None:
        _require_positive_id("song_id", self.song_id)
        _require_text("title", self.title)
        _require_text("artist", self.artist)
        _require_text("audio_path", self.audio_path)
        # El pipeline multimedia resuelve rutas relativas a la raíz del dataset y
        # usa '#' como separador del segmento.
        if (
            PurePosixPath(self.audio_path).is_absolute()
            or PureWindowsPath(self.audio_path).is_absolute()
            or "\\" in self.audio_path
            or "#" in self.audio_path
        ):
            raise ValueError("audio_path debe ser una ruta relativa POSIX sin '#'")
        duration = _require_finite("duration_seconds", self.duration_seconds)
        if duration <= 0:
            raise ValueError("duration_seconds debe ser > 0")
        object.__setattr__(self, "duration_seconds", duration)


@dataclass(frozen=True, slots=True)
class AudioSegment:
    segment_id: int
    song_id: int
    segment_index: int
    start_seconds: float
    end_seconds: float
    audio_ref: str

    def __post_init__(self) -> None:
        _require_positive_id("segment_id", self.segment_id)
        _require_positive_id("song_id", self.song_id)
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise ValueError(f"segment_index debe ser un entero >= 0, no {self.segment_index!r}")
        start = _require_finite("start_seconds", self.start_seconds)
        end = _require_finite("end_seconds", self.end_seconds)
        if start < 0:
            raise ValueError("start_seconds debe ser >= 0")
        if end <= start:
            raise ValueError("end_seconds debe ser > start_seconds")
        _require_text("audio_ref", self.audio_ref)
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)


class CopyrightCatalog:
    """Acceso a ``songs`` y ``audio_segments`` sobre un ``Catalog`` de SoupDB.

    El ``Catalog`` lo abre y lo cierra quien lo crea (con la ruta de la base); esta
    clase no lo cierra. Llamar a :meth:`initialize` antes de usarla.
    """

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        # Reentrante: las escrituras comprueban invariantes llamando a las lecturas.
        self._lock = threading.RLock()

    # --- bootstrap --------------------------------------------------------

    def initialize(self) -> None:
        """Crea las tablas que falten y valida las existentes. Idempotente.

        Nunca borra datos. Si una tabla existe con otro schema lanza
        :class:`IncompatibleSchemaError` en vez de seguir sobre datos que no
        corresponden.
        """
        with self._lock:
            for table, schema in ((SONGS, SONGS_SCHEMA), (AUDIO_SEGMENTS, AUDIO_SEGMENTS_SCHEMA)):
                if table in self._catalog.tables():
                    actual = self._catalog.schema(table)
                    if actual != schema:
                        raise IncompatibleSchemaError(
                            f"la tabla {table!r} existe con un schema incompatible: "
                            f"esperado {_describe(schema)}, encontrado {_describe(actual)}"
                        )
                else:
                    self._run(f"CREATE TABLE {table} ({_describe(schema)})")
                for column in _INDEXED_COLUMNS[table]:
                    if not self._catalog.indexes_for(table, column):
                        self._run(f"CREATE INDEX {table}_{column} ON {table} ({column}) TYPE BTREE")

    def has_vector_index(self) -> bool:
        """Si ``audio_segments`` tiene un índice vectorial (ranked) registrado.

        Con uno, el motor bloquea ``INSERT``/``DELETE`` en esa tabla fuera de una
        ``TransactionalSession``: la ingesta lo consulta para fallar antes de
        escribir la canción en vez de dejarla sin segmentos.
        """
        return bool(self._catalog.ranked_metadata_for(AUDIO_SEGMENTS))

    # --- songs ------------------------------------------------------------

    def add_song(self, song: Song) -> None:
        with self._lock:
            if self.get_song(song.song_id) is not None:
                raise IntegrityError(f"song_id {song.song_id} ya existe")
            self._run(f"INSERT INTO {SONGS} VALUES ({_sql_values(song)})")

    def get_song(self, song_id: int) -> Song | None:
        _require_positive_id("song_id", song_id)
        rows = self._run(f"SELECT * FROM {SONGS} WHERE song_id = {_sql_literal(song_id)}").rows
        return _single(rows, Song, "song_id", song_id)

    def list_songs(self) -> list[Song]:
        rows = self._run(f"SELECT * FROM {SONGS} ORDER BY song_id").rows
        return [Song(*row) for row in rows]

    # --- audio_segments ---------------------------------------------------

    def add_segment(self, segment: AudioSegment) -> None:
        with self._lock:
            song = self.get_song(segment.song_id)
            if song is None:
                raise IntegrityError(f"song_id {segment.song_id} no existe")
            expected_ref = segment_audio_ref(song.audio_path, segment.segment_index)
            if segment.audio_ref != expected_ref:
                raise IntegrityError(
                    f"audio_ref {segment.audio_ref!r} no corresponde al segmento; "
                    f"se esperaba {expected_ref!r}"
                )
            if self.get_segment(segment.segment_id) is not None:
                raise IntegrityError(f"segment_id {segment.segment_id} ya existe")
            repeated = self._run(
                f"SELECT segment_id FROM {AUDIO_SEGMENTS} "
                f"WHERE song_id = {_sql_literal(segment.song_id)} "
                f"AND segment_index = {_sql_literal(segment.segment_index)}"
            ).rows
            if repeated:
                raise IntegrityError(
                    f"la canción {segment.song_id} ya tiene el segment_index "
                    f"{segment.segment_index}"
                )
            self._run(f"INSERT INTO {AUDIO_SEGMENTS} VALUES ({_sql_values(segment)})")

    def get_segment(self, segment_id: int) -> AudioSegment | None:
        _require_positive_id("segment_id", segment_id)
        rows = self._run(
            f"SELECT * FROM {AUDIO_SEGMENTS} WHERE segment_id = {_sql_literal(segment_id)}"
        ).rows
        return _single(rows, AudioSegment, "segment_id", segment_id)

    def list_segments(self, song_id: int) -> list[AudioSegment]:
        _require_positive_id("song_id", song_id)
        rows = self._run(
            f"SELECT * FROM {AUDIO_SEGMENTS} WHERE song_id = {_sql_literal(song_id)} "
            "ORDER BY segment_index"
        ).rows
        return [AudioSegment(*row) for row in rows]

    # --- internals --------------------------------------------------------

    def _run(self, sql: str) -> ResultSet:
        # Lecturas y escrituras pasan por el mismo lock: no se intercalan entre hilos.
        with self._lock:
            return execute_sql(sql, self._catalog)


def _sql_literal(value: str | int | float) -> str:
    """Literal SQL de SoupDB para un valor ya validado por los modelos.

    Solo ``str``, ``int`` y ``float`` exactos (``type(...) is``): un ``bool``, una
    subclase o un texto nunca se cuelan como número.
    """
    if type(value) is str:
        return "'" + value.replace("'", "''") + "'"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"no se puede serializar el float {value!r}")
        # El lexer no admite exponentes (1e-05) y trata como INT un número sin
        # punto; Decimal(repr(x)) da el decimal más corto que reproduce el float.
        text = format(Decimal(repr(value)), "f")
        return text if "." in text else f"{text}.0"
    raise TypeError(f"tipo no admitido en SQL: {type(value).__name__}")


def _sql_values(record: Song | AudioSegment) -> str:
    return ", ".join(_sql_literal(value) for value in astuple(record))


def _single(rows, model, key: str, value: int):
    if not rows:
        return None
    if len(rows) > 1:
        raise IntegrityError(f"{key} {value} está duplicado en la base")
    return model(*rows[0])


def _describe(schema: tuple[ColumnDef, ...]) -> str:
    return ", ".join(f"{c.name} {c.type_name.value}" for c in schema)
