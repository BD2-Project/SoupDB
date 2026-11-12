"""Índice HNSW real de SoupDB sobre ``audio_segments.audio_ref``.

Flujo, reutilizando el pipeline multimedia de la Parte 4 (``soupdb-vector``):

1. :func:`write_training_manifest`: manifest ``audio_segment`` generado desde el
   catálogo relacional (artefacto de construcción, no fuente de verdad).
2. :func:`train_codebook`: MFCC → K-Means (K=128) → IDF con
   ``build_vectors.train_from_manifest``; se guarda una vez en ``codebook.npz``.
3. :func:`open_catalog`: abre el ``Catalog`` con un ``OnDiskVectorFactory`` y el
   perfil ``audio_segment`` (dataset, codebook y caché de descriptores).
4. :func:`ensure_index`: ``CREATE INDEX ... USING HNSW`` por SQL. La factoría
   vectoriza cada fila (MFCC → BoW TF-IDF L2) y guarda en HNSW el RID real de la
   fila; las consultas ``SIMILAR_TO`` resuelven cada hit a su fila relacional.

No genera ``vectores.bin``: la factoría SQL vectoriza las filas directamente.
No hay todavía detección de un archivo completo, agregación temporal ni umbral.

Orden obligatorio: ingestar todo el catálogo, entrenar el codebook y después
crear el índice. Con el índice creado, el motor bloquea ``INSERT``/``DELETE`` en
``audio_segments`` hechos con ``execute_sql`` (exige ``TransactionalSession`` con
deshacer reversible), así que ``CopyrightCatalog``/``ingest_song`` ya no pueden
añadir canciones; el bloqueo ocurre al planificar, antes de escribir nada.
Además la base debe abrirse siempre con :func:`open_catalog`: sin la factoría el
índice no se carga y las consultas vectoriales fallan.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from soupdb_vector.indexes.sql_factory import OnDiskVectorFactory
from soupdb_vector.multimedia.build_vectors import DEFAULT_MAX_DESCRIPTORS, train_from_manifest
from soupdb_vector.multimedia.codebook import save_codebook
from soupdb_vector.multimedia.manifest import write_manifest
from soupdb_vector.multimedia.sql_profiles import MultimediaProfile

from applications.copyright_detection.catalog import AUDIO_SEGMENTS, CopyrightCatalog
from applications.copyright_detection.config import INDEX_KIND, METRIC
from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.query import execute_sql

PROFILE_ID = "copyright-audio-v1"
INDEX_NAME = "copyright_audio_hnsw"
INDEXED_COLUMN = "audio_ref"
#: Palabras acústicas del codebook = dimensión de los vectores TF-IDF.
CODEBOOK_K = 128
VECTOR_DIMENSION = CODEBOOK_K


class CopyrightIndexError(Exception):
    """El índice vectorial de copyright no se pudo preparar o reabrir."""


@dataclass(frozen=True, slots=True)
class CopyrightIndexConfig:
    """Rutas del índice; todas las da el caller.

    - ``dataset_root``: raíz a la que son relativos ``audio_path``/``audio_ref``.
    - ``codebook_path``: ``codebook.npz`` del perfil (debe terminar en ``.npz``,
      ``numpy.savez`` añadiría la extensión).
    - ``descriptor_cache``: caché de MFCC por segmento, compartida por el
      entrenamiento y la vectorización.
    - ``vector_storage``: carpeta donde ``OnDiskVectorFactory`` guarda el HNSW.
    """

    dataset_root: Path
    codebook_path: Path
    descriptor_cache: Path
    vector_storage: Path
    profile_id: str = PROFILE_ID

    def __post_init__(self) -> None:
        for name in ("dataset_root", "codebook_path", "descriptor_cache", "vector_storage"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.codebook_path.suffix != ".npz":
            raise ValueError(f"codebook_path debe terminar en .npz: {self.codebook_path}")
        if not self.profile_id.strip():
            raise ValueError("profile_id no puede estar vacío")

    def profile(self) -> MultimediaProfile:
        return MultimediaProfile(
            self.profile_id,
            "audio_segment",
            self.dataset_root,
            self.codebook_path,
            self.descriptor_cache,
        )

    def vector_factory(self) -> OnDiskVectorFactory:
        return OnDiskVectorFactory(self.vector_storage, {self.profile_id: self.profile()})


def create_index_sql(profile_id: str = PROFILE_ID) -> str:
    # El parser solo admite PROFILE como literal de texto; se escapa la comilla.
    profile = profile_id.replace("'", "''")
    return (
        f"CREATE INDEX {INDEX_NAME} ON {AUDIO_SEGMENTS} ({INDEXED_COLUMN}) "
        f"USING {INDEX_KIND} "
        f"WITH METRIC={METRIC}, DIMENSION={VECTOR_DIMENSION}, PROFILE='{profile}'"
    )


def write_training_manifest(catalog: CopyrightCatalog, path: str | Path) -> int:
    """Escribe el manifest ``audio_segment`` del catálogo, ordenado por ``segment_id``.

    Mismo formato que ``soupdb_vector.multimedia.manifest.audio_rows``: rutas
    relativas a la raíz del dataset y una fila por segmento registrado. Devuelve
    el número de filas; falla si el catálogo no tiene segmentos.
    """
    songs = {song.song_id: song for song in catalog.list_songs()}
    segments = sorted(
        (segment for song_id in songs for segment in catalog.list_segments(song_id)),
        key=lambda segment: segment.segment_id,
    )
    if not segments:
        raise CopyrightIndexError("el catálogo no tiene segmentos de audio que indexar")
    rows = (
        {
            "id": segment.segment_id,
            "type": "audio_segment",
            "path": songs[segment.song_id].audio_path,
            "segment": segment.segment_index,
            "start": round(segment.start_seconds, 3),
            "end": round(segment.end_seconds, 3),
        }
        for segment in segments
    )
    return write_manifest(path, rows)


def train_codebook(
    catalog: CopyrightCatalog,
    config: CopyrightIndexConfig,
    manifest_path: str | Path,
    *,
    seed: int = 0,
    max_descriptors: int | None = DEFAULT_MAX_DESCRIPTORS,
) -> int:
    """Entrena el codebook K=128 con los segmentos del catálogo y lo guarda.

    No sobrescribe un codebook existente: el índice guarda su huella y otro
    codebook lo invalidaría. Devuelve el número de segmentos usados.
    """
    if config.codebook_path.exists():
        raise CopyrightIndexError(
            f"el codebook ya existe ({config.codebook_path}); no se sobrescribe"
        )
    if not config.dataset_root.is_dir():
        raise CopyrightIndexError(f"la raíz del dataset no es un directorio: {config.dataset_root}")
    count = write_training_manifest(catalog, manifest_path)
    try:
        centroids, idf = train_from_manifest(
            manifest_path,
            config.dataset_root,
            config.descriptor_cache,
            CODEBOOK_K,
            max_descriptors,
            seed,
        )
    except ValueError as exc:
        raise CopyrightIndexError(f"no se pudo entrenar el codebook K={CODEBOOK_K}: {exc}") from exc
    if centroids.shape[0] != CODEBOOK_K or idf.shape != (CODEBOOK_K,):
        raise CopyrightIndexError(f"codebook inesperado: {centroids.shape}, idf {idf.shape}")

    config.codebook_path.parent.mkdir(parents=True, exist_ok=True)
    staging = config.codebook_path.with_name(config.codebook_path.stem + ".partial.npz")
    save_codebook(staging, centroids, idf)
    os.replace(staging, config.codebook_path)
    return count


def open_catalog(db_path: str | Path, config: CopyrightIndexConfig) -> Catalog:
    """Abre la base de copyright con la factoría HNSW y el perfil de ``config``.

    Comprueba antes que el dataset y el codebook existen y que el codebook da
    vectores de dimensión 128. Si ya hay un índice, el ``Catalog`` lo reabre y la
    factoría verifica que perfil, raíz y codebook no cambiaron.
    """
    if not config.dataset_root.is_dir():
        raise CopyrightIndexError(f"la raíz del dataset no es un directorio: {config.dataset_root}")
    if not config.codebook_path.is_file():
        raise CopyrightIndexError(f"no existe el codebook {config.codebook_path}")
    factory = config.vector_factory()
    try:
        dimension = factory.query_vectorizer.dimension(config.profile_id)
    except QueryExecutionError as exc:
        raise CopyrightIndexError(f"codebook inválido para {config.profile_id!r}: {exc}") from exc
    if dimension != VECTOR_DIMENSION:
        raise CopyrightIndexError(
            f"el codebook da vectores de dimensión {dimension}; se espera {VECTOR_DIMENSION}"
        )
    try:
        return Catalog(Path(db_path), ranked_factory=factory)
    except QueryExecutionError as exc:
        raise CopyrightIndexError(
            f"no se pudo abrir la base con el perfil {config.profile_id!r}: {exc}"
        ) from exc


def ensure_index(catalog: Catalog, config: CopyrightIndexConfig) -> bool:
    """Crea el HNSW de copyright si no existe; devuelve si lo creó.

    Si ya existe se valida y se reutiliza (no se reconstruye). Si existe con otra
    configuración, o no está cargado porque el ``Catalog`` se abrió sin la
    factoría, falla sin tocarlo.
    """
    copyright_catalog = CopyrightCatalog(catalog)
    copyright_catalog.initialize()

    existing = catalog.ranked_metadata_for(AUDIO_SEGMENTS).get(INDEX_NAME)
    if existing is not None:
        expected = (AUDIO_SEGMENTS, INDEXED_COLUMN, INDEX_KIND, METRIC, VECTOR_DIMENSION)
        actual = (existing.table, existing.column, existing.algorithm, existing.metric)
        actual += (existing.dimension,)
        if actual != expected or existing.profile_id != config.profile_id:
            raise CopyrightIndexError(
                f"el índice {INDEX_NAME!r} ya existe con otra configuración: {existing}"
            )
        if INDEX_NAME not in catalog.ranked_indexes_for(AUDIO_SEGMENTS):
            raise CopyrightIndexError(
                f"el índice {INDEX_NAME!r} existe pero no está cargado; abrir con open_catalog"
            )
        return False

    if not any(copyright_catalog.list_segments(s.song_id) for s in copyright_catalog.list_songs()):
        raise CopyrightIndexError("el catálogo no tiene segmentos de audio que indexar")
    try:
        execute_sql(create_index_sql(config.profile_id), catalog)
    except QueryExecutionError as exc:
        raise CopyrightIndexError(f"no se pudo crear {INDEX_NAME!r}: {exc}") from exc
    return True
