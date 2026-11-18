"""Índice HNSW real de copyright (issue #176): codebook K=128, SQL y persistencia.

Sin mocks: WAV sintéticos deterministas, MFCC/K-Means/TF-IDF del plugin, HNSW en
disco con ``OnDiskVectorFactory`` y consultas ``SIMILAR_TO`` por SQL.
"""

import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from soupdb_vector.multimedia.audio_features import SAMPLE_RATE
from soupdb_vector.multimedia.build_vectors import validate_manifest
from soupdb_vector.multimedia.codebook import load_codebook, save_codebook
from soupdb_vector.multimedia.manifest import read_manifest

from applications.copyright_detection.catalog import AUDIO_SEGMENTS, CopyrightCatalog
from applications.copyright_detection.indexing import (
    CODEBOOK_K,
    INDEX_NAME,
    PROFILE_ID,
    VECTOR_DIMENSION,
    CopyrightIndexConfig,
    CopyrightIndexError,
    create_index_sql,
    ensure_index,
    open_catalog,
    train_codebook,
    write_training_manifest,
)
from applications.copyright_detection.ingestion import (
    AudioIngestionError,
    SongMetadata,
    ingest_song,
)
from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.query import execute_sql

#: (song_id, audio_path, segundos, primer segment_id)
SONGS = (
    (1, "tracks/a.wav", 13.0, 1),
    (2, "tracks/b.wav", 11.0, 101),
    (3, "tracks/c.wav", 8.0, 201),
)

QUERY = (
    "SELECT segment_id, song_id, audio_ref, SCORE() AS score FROM audio_segments "
    "WHERE SIMILAR_TO('{ref}', k={k}) USING HNSW WITH METRIC=euclidean "
    "ORDER BY score DESC"
)


def write_track(path: Path, seconds: float, seed: int) -> None:
    """Pista reproducible: cada bloque de 5 s con sus propias frecuencias + ruido leve."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = round(seconds * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    signal = np.zeros(samples)
    block = round(5.0 * SAMPLE_RATE)
    for start in range(0, samples, block):
        piece = slice(start, start + block)
        for frequency in rng.uniform(150, 4000, size=3):
            signal[piece] += 0.2 * np.sin(2 * np.pi * frequency * t[piece])
    signal += 0.02 * rng.standard_normal(samples)
    sf.write(path, signal, SAMPLE_RATE)


def ingest_all(db: CopyrightCatalog, root: Path, songs=SONGS) -> None:
    for song_id, rel, seconds, first in songs:
        write_track(root / rel, seconds, seed=song_id)
        ingest_song(db, root, SongMetadata(song_id, f"t{song_id}", "a", rel), first)


@dataclass(frozen=True)
class Built:
    root: Path
    db: Path
    codebook: Path
    cache: Path
    manifest: Path
    segment_count: int


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Built:
    """Catálogo ingestado y codebook K=128 entrenado una vez para todo el módulo."""
    base = tmp_path_factory.mktemp("copyright_index")
    root, db_path = base / "dataset", base / "db"
    root.mkdir()
    catalog = Catalog(db_path)
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        ingest_all(db, root)
        config = CopyrightIndexConfig(
            root, base / "codebook.npz", base / "cache", base / "unused-vectors"
        )
        count = train_codebook(db, config, base / "train.jsonl")
    finally:
        catalog.close()
    return Built(root, db_path, config.codebook_path, base / "cache", base / "train.jsonl", count)


@pytest.fixture
def config(built, tmp_path) -> CopyrightIndexConfig:
    # Base propia por test (se copia); codebook y caché compartidos (solo lectura).
    shutil.copytree(built.db, tmp_path / "db")
    return CopyrightIndexConfig(built.root, built.codebook, built.cache, tmp_path / "vectors")


@pytest.fixture
def index_built(config, tmp_path) -> Path:
    """Crea el HNSW en la base del test y la cierra; devuelve la ruta de la base."""
    catalog = open_catalog(tmp_path / "db", config)
    try:
        assert ensure_index(catalog, config) is True
    finally:
        catalog.close()
    return tmp_path / "db"


@pytest.fixture
def indexed(index_built, config):
    catalog = open_catalog(index_built, config)
    try:
        yield catalog
    finally:
        catalog.close()


def query(catalog: Catalog, ref: str, k: int = 3):
    return execute_sql(QUERY.format(ref=ref, k=k), catalog).rows


# --- manifest y codebook ----------------------------------------------------


def test_training_manifest_is_deterministic_relative_and_sorted(built, tmp_path):
    shutil.copytree(built.db, tmp_path / "db-copy")
    catalog = Catalog(tmp_path / "db-copy")
    try:
        db = CopyrightCatalog(catalog)
        first = write_training_manifest(db, tmp_path / "m1.jsonl")
        second = write_training_manifest(db, tmp_path / "m2.jsonl")
    finally:
        catalog.close()

    assert first == second == built.segment_count == 7  # 3 + 2 + 2 segmentos
    assert (tmp_path / "m1.jsonl").read_bytes() == (tmp_path / "m2.jsonl").read_bytes()
    assert (tmp_path / "m1.jsonl").read_bytes() == built.manifest.read_bytes()
    assert validate_manifest(built.manifest) == ("audio_segment", 7)
    rows = list(read_manifest(built.manifest))
    assert [row["id"] for row in rows] == [1, 2, 3, 101, 102, 201, 202]
    assert all(not Path(row["path"]).is_absolute() for row in rows)
    assert rows[0] == {
        "id": 1,
        "type": "audio_segment",
        "path": "tracks/a.wav",
        "segment": 0,
        "start": 0.0,
        "end": 5.0,
    }


def test_codebook_has_128_words_of_mfcc_dimension(built):
    centroids, idf = load_codebook(built.codebook)

    assert CODEBOOK_K == VECTOR_DIMENSION == 128
    assert centroids.shape == (128, 20)
    assert idf.shape == (128,)
    assert np.isfinite(centroids).all() and np.isfinite(idf).all()


def test_train_refuses_to_overwrite_existing_codebook(built, tmp_path):
    before = built.codebook.read_bytes()
    catalog = Catalog(tmp_path / "db")
    try:
        config = CopyrightIndexConfig(built.root, built.codebook, built.cache, tmp_path / "v")
        with pytest.raises(CopyrightIndexError, match="ya existe"):
            train_codebook(CopyrightCatalog(catalog), config, tmp_path / "m.jsonl")
    finally:
        catalog.close()
    assert built.codebook.read_bytes() == before


def test_train_rejects_empty_catalog(built, tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        config = CopyrightIndexConfig(
            built.root, tmp_path / "cb.npz", tmp_path / "c", tmp_path / "v"
        )
        with pytest.raises(CopyrightIndexError, match="no tiene segmentos"):
            train_codebook(db, config, tmp_path / "m.jsonl")
    finally:
        catalog.close()
    assert not (tmp_path / "cb.npz").exists()


def test_config_requires_npz_codebook(tmp_path):
    with pytest.raises(ValueError, match=".npz"):
        CopyrightIndexConfig(tmp_path, tmp_path / "codebook", tmp_path / "c", tmp_path / "v")


def test_create_index_sql_uses_parser_syntax():
    assert create_index_sql() == (
        "CREATE INDEX copyright_audio_hnsw ON audio_segments (audio_ref) USING HNSW "
        "WITH METRIC=euclidean, DIMENSION=128, PROFILE='copyright-audio-v1'"
    )


# --- creación y consulta ----------------------------------------------------


def test_ensure_index_creates_real_hnsw_with_expected_metadata(indexed, config, built):
    meta = indexed.ranked_metadata_for(AUDIO_SEGMENTS)[INDEX_NAME]
    assert (meta.column, meta.algorithm, meta.metric, meta.dimension, meta.profile_id) == (
        "audio_ref",
        "HNSW",
        "euclidean",
        128,
        PROFILE_ID,
    )
    binding = indexed.ranked_indexes_for(AUDIO_SEGMENTS)[INDEX_NAME]
    assert binding.item_count == built.segment_count

    manifests = list(config.vector_storage.glob("*/sql_index.json"))
    assert len(manifests) == 1
    stored = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert (stored["algorithm"], stored["metric"], stored["dimension"]) == (
        "HNSW",
        "euclidean",
        128,
    )
    assert stored["kind"] == "audio_segment" and stored["count"] == built.segment_count
    assert not list(config.vector_storage.rglob("*.bin"))  # sin vectores.bin


def test_similar_to_resolves_hits_to_relational_rows(indexed):
    rows = query(indexed, "tracks/a.wav#segment=000001", k=3)

    assert len(rows) == 3
    # El propio segmento es el más parecido (distancia 0 → score 1.0).
    assert rows[0][:3] == (2, 1, "tracks/a.wav#segment=000001")
    assert rows[0][3] == pytest.approx(1.0)
    scores = [row[3] for row in rows]
    assert all(type(score) is float and 0 < score <= 1 for score in scores)
    assert scores == sorted(scores, reverse=True)
    # Cada hit (RID) es una fila real: segment_id, song_id y audio_ref coinciden con el catálogo.
    db = CopyrightCatalog(indexed)
    for segment_id, song_id, audio_ref, _ in rows:
        segment = db.get_segment(segment_id)
        assert (segment.song_id, segment.audio_ref) == (song_id, audio_ref)


def test_every_catalog_segment_finds_itself_first(indexed):
    db = CopyrightCatalog(indexed)
    for song in db.list_songs():
        for segment in db.list_segments(song.song_id):
            assert query(indexed, segment.audio_ref, k=1)[0][0] == segment.segment_id


def test_ensure_index_twice_reuses_without_rebuilding(indexed, config):
    manifest = next(config.vector_storage.glob("*/sql_index.json"))
    before = manifest.stat().st_mtime_ns

    assert ensure_index(indexed, config) is False
    assert manifest.stat().st_mtime_ns == before
    with pytest.raises(QueryExecutionError, match="already exists"):
        execute_sql(create_index_sql(), indexed)


def test_ingest_after_index_fails_before_writing(indexed, tmp_path):
    # Con el HNSW creado, el motor exige TransactionalSession para escribir en
    # audio_segments; la ingesta lo detecta antes de insertar la canción.
    root = tmp_path / "extra"
    write_track(root / "tracks/new.wav", 6.0, seed=99)
    db = CopyrightCatalog(indexed)
    before = db.list_songs()
    assert db.has_vector_index() is True

    with pytest.raises(AudioIngestionError, match="ya tiene índice vectorial"):
        ingest_song(db, root, SongMetadata(9, "t", "a", "tracks/new.wav"), 900)

    assert db.list_songs() == before
    assert db.get_song(9) is None and db.get_segment(900) is None
    assert len(query(indexed, "tracks/a.wav#segment=000000", k=10)) == 7


# --- persistencia -----------------------------------------------------------


def test_index_persists_and_reopens_without_rebuild(config, tmp_path):
    catalog = open_catalog(tmp_path / "db", config)
    try:
        ensure_index(catalog, config)
        first = query(catalog, "tracks/b.wav#segment=000000", k=4)
    finally:
        catalog.close()
    manifest = next(config.vector_storage.glob("*/sql_index.json"))
    before = manifest.stat().st_mtime_ns

    # Perfil y factoría nuevos con la misma configuración.
    reopened_config = CopyrightIndexConfig(
        config.dataset_root, config.codebook_path, config.descriptor_cache, config.vector_storage
    )
    catalog = open_catalog(tmp_path / "db", reopened_config)
    try:
        assert ensure_index(catalog, reopened_config) is False
        assert query(catalog, "tracks/b.wav#segment=000000", k=4) == first
        assert first[0][:2] == (101, 2)
    finally:
        catalog.close()
    assert manifest.stat().st_mtime_ns == before


# --- errores ----------------------------------------------------------------


def test_open_without_codebook_fails(config, tmp_path):
    missing = replace(config, codebook_path=tmp_path / "missing.npz")

    with pytest.raises(CopyrightIndexError, match="no existe el codebook"):
        open_catalog(tmp_path / "db", missing)


def test_open_with_wrong_dimension_codebook_fails(config, tmp_path):
    other = tmp_path / "k64.npz"
    save_codebook(other, np.zeros((64, 20), dtype=np.float32), np.ones(64))

    with pytest.raises(CopyrightIndexError, match="dimensión 64"):
        open_catalog(tmp_path / "db", replace(config, codebook_path=other))


def test_open_with_wrong_dataset_root_fails(config, tmp_path):
    with pytest.raises(CopyrightIndexError, match="raíz del dataset"):
        open_catalog(tmp_path / "db", replace(config, dataset_root=tmp_path / "nope"))


def test_reopen_with_other_profile_id_fails(index_built, config):
    with pytest.raises(CopyrightIndexError, match="unknown multimedia profile") as info:
        open_catalog(index_built, replace(config, profile_id="otro-perfil"))
    assert isinstance(info.value.__cause__, QueryExecutionError)


def test_reopen_with_changed_codebook_fails(index_built, config, tmp_path):
    other = tmp_path / "other.npz"
    centroids, idf = load_codebook(config.codebook_path)
    save_codebook(other, centroids + 1.0, idf)

    with pytest.raises(CopyrightIndexError, match="profile/configuration changed"):
        open_catalog(index_built, replace(config, codebook_path=other))


def test_reopen_with_other_dataset_root_fails(index_built, config, tmp_path):
    moved = tmp_path / "moved"
    shutil.copytree(config.dataset_root, moved)

    with pytest.raises(CopyrightIndexError, match="profile/configuration changed"):
        open_catalog(index_built, replace(config, dataset_root=moved))


def test_catalog_opened_without_factory_cannot_use_the_index(index_built, config):
    plain = Catalog(index_built)
    try:
        with pytest.raises(CopyrightIndexError, match="no está cargado"):
            ensure_index(plain, config)
    finally:
        plain.close()


def test_ensure_index_on_empty_catalog_fails(config, tmp_path):
    catalog = open_catalog(tmp_path / "empty-db", config)
    try:
        with pytest.raises(CopyrightIndexError, match="no tiene segmentos"):
            ensure_index(catalog, config)
        assert catalog.ranked_metadata_for(AUDIO_SEGMENTS) == {}
    finally:
        catalog.close()
