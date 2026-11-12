"""Búsqueda de un audio candidato contra el catálogo original, segmento a segmento.

El candidato se segmenta con ``iter_audio_segments`` (la misma regla de 5 s y de
último segmento que los originales). Por cada segmento se ejecuta por SQL una
consulta ``SIMILAR_TO ... USING HNSW``: el planner de SoupDB vectoriza la
referencia ``<audio_path>#segment=NNNNNN`` con el perfil del índice (MFCC →
codebook → BoW TF-IDF L2), pide los ``k`` vecinos al HNSW y resuelve cada RID a
su fila de ``audio_segments``. Después se añaden título y artista desde
``CopyrightCatalog``.

Esta capa **no decide copyright**: no hay umbral ni agregación temporal; solo
devuelve los ``k`` vecinos de cada segmento. ``score`` es la similitud del motor
para ``euclidean``, ``1 / (1 + distancia)``: mayor es más parecido, y no es una
probabilidad.

Limitaciones:

- El vectorizador del perfil solo resuelve rutas bajo ``dataset_root``, así que
  el candidato debe estar ahí (``audio_path`` relativo). El candidato no se
  inserta en SoupDB ni en el HNSW.
- La caché de descriptores del perfil usa como clave la ruta, no el contenido:
  si se reemplaza el archivo de un candidato manteniendo la ruta, se reutilizan
  los MFCC anteriores. Cada candidato distinto debe tener una ruta propia.
- El candidato se decodifica para segmentarlo y otra vez dentro del vectorizador
  (la primera consulta llena la caché con todos sus segmentos).
"""

from dataclasses import dataclass
from pathlib import Path

from soupdb_vector.multimedia.audio_features import (
    SAMPLE_RATE,
    iter_audio_segments,
    segment_id,
)

from applications.copyright_detection.catalog import AUDIO_SEGMENTS, CopyrightCatalog, Song
from applications.copyright_detection.config import INDEX_KIND, METRIC, SEGMENT_SECONDS
from applications.copyright_detection.indexing import (
    CopyrightIndexConfig,
    CopyrightIndexError,
    require_index,
)
from applications.copyright_detection.ingestion import (
    AUDIO_DECODE_ERRORS,
    AudioIngestionError,
    resolve_dataset_path,
)
from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.query import execute_sql

_QUERY = (
    "SELECT segment_id, song_id, segment_index, start_seconds, end_seconds, audio_ref, "
    "SCORE() AS score FROM {table} "
    "WHERE SIMILAR_TO({reference}, k={k}) USING {algorithm} WITH METRIC={metric} "
    "ORDER BY score DESC"
)


class CopyrightSearchError(Exception):
    """La búsqueda del candidato no se pudo realizar; el mensaje indica la causa."""


@dataclass(frozen=True, slots=True)
class SegmentMatch:
    """Un segmento original vecino de un segmento del candidato."""

    segment_id: int
    song_id: int
    title: str
    artist: str
    source_segment_index: int
    source_start_seconds: float
    source_end_seconds: float
    audio_ref: str
    #: Similitud del motor (euclidean: 1 / (1 + distancia)). No es una probabilidad.
    score: float


@dataclass(frozen=True, slots=True)
class QuerySegmentResult:
    """Vecinos de un segmento del candidato, de mayor a menor ``score``."""

    query_segment_index: int
    query_start_seconds: float
    query_end_seconds: float
    matches: tuple[SegmentMatch, ...]


@dataclass(frozen=True, slots=True)
class AudioSearchResult:
    """Resultado por segmento; los matches de segmentos distintos no se mezclan."""

    audio_path: str
    k: int
    segments: tuple[QuerySegmentResult, ...]


def search_audio(
    catalog: Catalog, config: CopyrightIndexConfig, audio_path: str, k: int
) -> AudioSearchResult:
    """Busca los ``k`` segmentos originales más parecidos a cada segmento del candidato.

    ``catalog`` debe venir de ``open_catalog(db_path, config)`` con el índice ya
    creado; esta función no lo crea ni lo reconstruye. ``audio_path`` es relativo
    a ``config.dataset_root``. No escribe nada en la base.
    """
    if type(k) is not int or k <= 0:
        raise CopyrightSearchError(f"k debe ser un entero positivo, no {k!r}")
    try:
        require_index(catalog, config.profile_id)
    except CopyrightIndexError as exc:
        raise CopyrightSearchError(f"índice de copyright no disponible: {exc}") from exc
    _check_profile_root(catalog, config)
    try:
        path = resolve_dataset_path(config.dataset_root, audio_path)
    except AudioIngestionError as exc:
        raise CopyrightSearchError(f"ruta de candidato inválida: {exc}") from exc

    try:
        pieces = [(index, len(samples)) for index, samples in iter_audio_segments(path)]
    except AUDIO_DECODE_ERRORS as exc:
        raise CopyrightSearchError(f"no se pudo decodificar {audio_path!r}: {exc}") from exc
    if not pieces:
        raise CopyrightSearchError(
            f"{audio_path!r} no produce ningún segmento de {SEGMENT_SECONDS} s; no se busca"
        )

    songs = _SongLookup(CopyrightCatalog(catalog))
    segments = []
    for index, length in pieces:
        start = index * SEGMENT_SECONDS
        rows = _similar_rows(catalog, segment_id(audio_path, index), k)
        segments.append(
            QuerySegmentResult(
                query_segment_index=index,
                query_start_seconds=start,
                query_end_seconds=start + length / SAMPLE_RATE,
                matches=tuple(songs.match(row) for row in rows),
            )
        )
    return AudioSearchResult(audio_path=audio_path, k=k, segments=tuple(segments))


def _check_profile_root(catalog: Catalog, config: CopyrightIndexConfig) -> None:
    # La ruta se valida contra config.dataset_root y el vectorizador la resuelve
    # contra la raíz de su perfil: deben ser la misma carpeta.
    vectorizer = catalog.query_vectorizer
    if vectorizer is None:
        raise CopyrightSearchError("el Catalog no tiene vectorizador; abrir con open_catalog")
    try:
        profile = vectorizer.profile(config.profile_id)
    except QueryExecutionError as exc:
        raise CopyrightSearchError(f"perfil {config.profile_id!r} no cargado: {exc}") from exc
    if Path(profile.root).resolve() != config.dataset_root.resolve():
        raise CopyrightSearchError(
            f"dataset_root {config.dataset_root} no es la raíz del perfil ({profile.root})"
        )


def _similar_rows(catalog: Catalog, reference: str, k: int) -> tuple[tuple[object, ...], ...]:
    sql = _QUERY.format(
        table=AUDIO_SEGMENTS,
        reference=_text_literal(reference),
        k=k,
        algorithm=INDEX_KIND,
        metric=METRIC,
    )
    try:
        return execute_sql(sql, catalog).rows
    except QueryExecutionError as exc:
        raise CopyrightSearchError(f"falló la búsqueda HNSW de {reference!r}: {exc}") from exc


def _text_literal(text: str) -> str:
    """Literal SQL de texto: comillas simples, ``'`` duplicada."""
    return "'" + text.replace("'", "''") + "'"


class _SongLookup:
    """Caché de canciones durante una búsqueda: una consulta por ``song_id``."""

    def __init__(self, catalog: CopyrightCatalog) -> None:
        self._catalog = catalog
        self._songs: dict[int, Song] = {}

    def song(self, song_id: int) -> Song:
        song = self._songs.get(song_id)
        if song is None:
            song = self._catalog.get_song(song_id)
            if song is None:
                raise CopyrightSearchError(
                    f"catálogo inconsistente: un segmento indexado apunta a song_id={song_id}, "
                    "que no existe en songs"
                )
            self._songs[song_id] = song
        return song

    def match(self, row: tuple[object, ...]) -> SegmentMatch:
        segment_id_, song_id, segment_index, start, end, audio_ref, score = row
        song = self.song(song_id)
        return SegmentMatch(
            segment_id=segment_id_,
            song_id=song_id,
            title=song.title,
            artist=song.artist,
            source_segment_index=segment_index,
            source_start_seconds=start,
            source_end_seconds=end,
            audio_ref=audio_ref,
            score=score,
        )
