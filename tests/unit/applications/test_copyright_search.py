"""Búsqueda HNSW por segmentos de un candidato (issue #178).

Sin mocks de la búsqueda: WAV sintéticos, catálogo e índice reales de las issues
anteriores y consultas SQL ``SIMILAR_TO`` reales. Solo se envuelve
``execute_sql`` para contar cuántas consultas se ejecutan (delegando en la real).
"""

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from soupdb_vector.multimedia.audio_features import SAMPLE_RATE

from applications.copyright_detection import search as search_module
from applications.copyright_detection.catalog import AUDIO_SEGMENTS, CopyrightCatalog
from applications.copyright_detection.indexing import (
    INDEX_NAME,
    CopyrightIndexConfig,
    ensure_index,
    open_catalog,
    train_codebook,
)
from applications.copyright_detection.ingestion import (
    AudioIngestionError,
    SongMetadata,
    ingest_song,
)
from applications.copyright_detection.search import (
    AudioSearchResult,
    CopyrightSearchError,
    QuerySegmentResult,
    SegmentMatch,
    search_audio,
)
from engine.common.catalog import Catalog
from engine.query import execute_sql

#: (song_id, title, artist, audio_path, segundos, primer segment_id, semilla)
ORIGINALS = (
    (1, "Alpha", "Ana", "tracks/a.wav", 13.0, 1, 1),
    (2, "Beta", "Bruno", "tracks/b.wav", 11.0, 101, 2),
    (3, "Gamma", "O'Connor", "tracks/c.wav", 8.0, 201, 3),
)
TOTAL_SEGMENTS = 7  # 3 + 2 + 2

#: Candidatos (ruta, segundos, semilla). La misma semilla y duración que un
#: original produce un audio idéntico a ese original.
CANDIDATES = (
    ("queries/copy_a.wav", 13.0, 1),
    ("queries/tail12.wav", 12.0, 2),
    ("queries/short.wav", 2.0, 4),
    ("queries/it's.wav", 8.0, 3),
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


@dataclass(frozen=True)
class Built:
    root: Path
    db: Path
    codebook: Path
    cache: Path


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Built:
    """Originales ingestados, candidatos escritos y codebook K=128, una vez por módulo."""
    base = tmp_path_factory.mktemp("copyright_search")
    root, db_path = base / "dataset", base / "db"
    root.mkdir()
    for rel, seconds, seed in CANDIDATES:
        write_track(root / rel, seconds, seed)
    catalog = Catalog(db_path)
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        for song_id, title, artist, rel, seconds, first, seed in ORIGINALS:
            write_track(root / rel, seconds, seed)
            ingest_song(db, root, SongMetadata(song_id, title, artist, rel), first)
        config = CopyrightIndexConfig(root, base / "codebook.npz", base / "cache", base / "v")
        train_codebook(db, config, base / "train.jsonl")
    finally:
        catalog.close()
    return Built(root, db_path, base / "codebook.npz", base / "cache")


@pytest.fixture
def config(built, tmp_path) -> CopyrightIndexConfig:
    shutil.copytree(built.db, tmp_path / "db")
    return CopyrightIndexConfig(built.root, built.codebook, built.cache, tmp_path / "vectors")


@pytest.fixture
def index_built(config, tmp_path) -> Path:
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


@pytest.fixture
def sql_log(monkeypatch) -> list[str]:
    """Registra cada SQL que ejecuta la búsqueda y delega en el execute_sql real."""
    log: list[str] = []
    real = search_module.execute_sql

    def counting(sql, catalog):
        log.append(sql)
        return real(sql, catalog)

    monkeypatch.setattr(search_module, "execute_sql", counting)
    return log


def intervals(result: AudioSearchResult) -> list[tuple[int, float, float]]:
    return [
        (s.query_segment_index, s.query_start_seconds, s.query_end_seconds) for s in result.segments
    ]


# --- coincidencias ------------------------------------------------------------


def test_identical_candidate_finds_its_original_segments_first(indexed, config):
    result = search_audio(indexed, config, "queries/copy_a.wav", k=3)

    assert isinstance(result, AudioSearchResult)
    assert (result.audio_path, result.k) == ("queries/copy_a.wav", 3)
    for query_segment in result.segments:
        assert isinstance(query_segment, QuerySegmentResult)
        assert len(query_segment.matches) == 3
        top = query_segment.matches[0]
        assert isinstance(top, SegmentMatch)
        assert (top.segment_id, top.song_id, top.source_segment_index) == (
            1 + query_segment.query_segment_index,
            1,
            query_segment.query_segment_index,
        )
        assert (top.title, top.artist) == ("Alpha", "Ana")
        assert (top.source_start_seconds, top.source_end_seconds) == (
            query_segment.query_start_seconds,
            query_segment.query_end_seconds,
        )
        assert top.score == pytest.approx(1.0, abs=1e-6)
        scores = [match.score for match in query_segment.matches]
        assert scores == sorted(scores, reverse=True)
        assert all(type(score) is float and 0 < score <= 1 for score in scores)


def test_matches_are_real_catalog_rows_with_song_metadata(indexed, config):
    # Comprobación de RID: cada hit es una fila real de audio_segments.
    result = search_audio(indexed, config, "tracks/b.wav", k=TOTAL_SEGMENTS)
    db = CopyrightCatalog(indexed)
    titles = {song_id: (title, artist) for song_id, title, artist, *_ in ORIGINALS}

    assert [s.matches[0].segment_id for s in result.segments] == [101, 102]
    seen_songs = set()
    for query_segment in result.segments:
        for match in query_segment.matches:
            row = db.get_segment(match.segment_id)
            assert (
                row.song_id,
                row.segment_index,
                row.start_seconds,
                row.end_seconds,
                row.audio_ref,
            ) == (
                match.song_id,
                match.source_segment_index,
                match.source_start_seconds,
                match.source_end_seconds,
                match.audio_ref,
            )
            assert (match.title, match.artist) == titles[match.song_id]
            seen_songs.add(match.song_id)
    assert seen_songs == {1, 2, 3}


# --- segmentación del candidato ----------------------------------------------------


def test_13s_candidate_runs_three_independent_hnsw_queries(indexed, config, sql_log):
    result = search_audio(indexed, config, "queries/copy_a.wav", k=2)

    assert intervals(result) == [(0, 0.0, 5.0), (1, 5.0, 10.0), (2, 10.0, 13.0)]
    assert len(sql_log) == 3
    for index, sql in enumerate(sql_log):
        assert f"SIMILAR_TO('queries/copy_a.wav#segment={index:06d}', k=2)" in sql
        assert "USING HNSW WITH METRIC=euclidean ORDER BY score DESC" in sql
    # Cada segmento conserva su propia lista: los mejores son distintos.
    assert [s.matches[0].segment_id for s in result.segments] == [1, 2, 3]


def test_12s_candidate_drops_short_tail(indexed, config, sql_log):
    result = search_audio(indexed, config, "queries/tail12.wav", k=1)

    assert intervals(result) == [(0, 0.0, 5.0), (1, 5.0, 10.0)]
    assert len(sql_log) == 2


def test_too_short_candidate_fails_before_any_query(indexed, config, sql_log):
    with pytest.raises(CopyrightSearchError, match="ningún segmento"):
        search_audio(indexed, config, "queries/short.wav", k=3)
    assert sql_log == []


def test_undecodable_candidate_fails(indexed, config, tmp_path, sql_log):
    bad = config.dataset_root / f"queries/bad-{tmp_path.name}.wav"
    bad.write_bytes(b"esto no es audio" * 64)
    try:
        with pytest.raises(CopyrightSearchError, match="no se pudo decodificar"):
            search_audio(indexed, config, bad.relative_to(config.dataset_root).as_posix(), 1)
    finally:
        bad.unlink()
    assert sql_log == []


# --- k ----------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, True, 1.0, "3", None])
def test_invalid_k_is_rejected_before_any_query(indexed, config, sql_log, k):
    with pytest.raises(CopyrightSearchError, match="k debe ser"):
        search_audio(indexed, config, "queries/copy_a.wav", k)
    assert sql_log == []


def test_k_larger_than_catalog_returns_available_hits(indexed, config):
    result = search_audio(indexed, config, "queries/copy_a.wav", k=50)

    assert result.k == 50
    assert [len(s.matches) for s in result.segments] == [TOTAL_SEGMENTS] * 3
    for query_segment in result.segments:
        ids = [match.segment_id for match in query_segment.matches]
        assert len(set(ids)) == len(ids) == TOTAL_SEGMENTS


# --- rutas y escaping -----------------------------------------------------------


def test_reference_with_quote_is_escaped(indexed, config, sql_log):
    result = search_audio(indexed, config, "queries/it's.wav", k=2)

    assert intervals(result) == [(0, 0.0, 5.0), (1, 5.0, 8.0)]
    assert "SIMILAR_TO('queries/it''s.wav#segment=000000', k=2)" in sql_log[0]
    assert [s.matches[0].segment_id for s in result.segments] == [201, 202]
    assert result.segments[0].matches[0].artist == "O'Connor"


@pytest.mark.parametrize(
    "path",
    [
        "/queries/copy_a.wav",
        "../outside.wav",
        "queries/missing.wav",
        "queries/copy_a.wav#segment=000000",
        "queries",
        "queries/./copy_a.wav",
    ],
)
def test_invalid_candidate_path_is_rejected(indexed, config, sql_log, path):
    with pytest.raises(CopyrightSearchError, match="ruta de candidato inválida") as info:
        search_audio(indexed, config, path, k=1)
    assert isinstance(info.value.__cause__, AudioIngestionError)
    assert sql_log == []


def test_dataset_root_must_match_profile_root(indexed, config, tmp_path):
    with pytest.raises(CopyrightSearchError, match="raíz del perfil"):
        search_audio(indexed, replace(config, dataset_root=tmp_path), "queries/copy_a.wav", 1)


# --- índice -----------------------------------------------------------------


def test_missing_index_is_reported(config, tmp_path, sql_log):
    catalog = open_catalog(tmp_path / "db", config)  # sin ensure_index
    try:
        with pytest.raises(CopyrightSearchError, match="no existe el índice"):
            search_audio(catalog, config, "queries/copy_a.wav", k=1)
        assert catalog.ranked_metadata_for(AUDIO_SEGMENTS) == {}  # no se creó
    finally:
        catalog.close()
    assert sql_log == []


def test_catalog_without_vector_factory_is_reported(index_built, config):
    plain = Catalog(index_built)
    try:
        with pytest.raises(CopyrightSearchError, match="no está cargado"):
            search_audio(plain, config, "queries/copy_a.wav", k=1)
    finally:
        plain.close()


# --- no modifica / persistencia / inconsistencia --------------------------------------


def test_search_does_not_modify_catalog_or_index(indexed, config):
    db = CopyrightCatalog(indexed)
    binding = indexed.ranked_indexes_for(AUDIO_SEGMENTS)[INDEX_NAME]
    songs_before = db.list_songs()
    rows_before = execute_sql("SELECT * FROM audio_segments", indexed).rows
    vectors_before = len(binding.index)

    search_audio(indexed, config, "queries/copy_a.wav", k=3)
    search_audio(indexed, config, "queries/it's.wav", k=3)

    assert db.list_songs() == songs_before
    assert execute_sql("SELECT * FROM audio_segments", indexed).rows == rows_before
    assert len(binding.index) == vectors_before == TOTAL_SEGMENTS
    assert indexed.ranked_indexes_for(AUDIO_SEGMENTS)[INDEX_NAME].item_count == TOTAL_SEGMENTS
    assert not [r for r in rows_before if str(r[5]).startswith("queries/")]


def test_search_after_close_and_reopen(index_built, config):
    catalog = open_catalog(index_built, config)
    try:
        first = search_audio(catalog, config, "queries/copy_a.wav", k=3)
    finally:
        catalog.close()

    reopened = CopyrightIndexConfig(
        config.dataset_root, config.codebook_path, config.descriptor_cache, config.vector_storage
    )
    catalog = open_catalog(index_built, reopened)
    try:
        assert search_audio(catalog, reopened, "queries/copy_a.wav", k=3) == first
    finally:
        catalog.close()


def test_hit_pointing_to_missing_song_is_reported(config, tmp_path):
    # Fila huérfana insertada con SQL directo antes de indexar (CopyrightCatalog no
    # lo permitiría); el HNSW la incluye y la búsqueda debe detectarla.
    plain = Catalog(tmp_path / "db")
    try:
        execute_sql(
            "INSERT INTO audio_segments VALUES "
            "(999, 77, 0, 0.0, 5.0, 'tracks/c.wav#segment=000000')",
            plain,
        )
    finally:
        plain.close()
    catalog = open_catalog(tmp_path / "db", config)
    try:
        ensure_index(catalog, config)
        with pytest.raises(CopyrightSearchError, match="song_id=77"):
            search_audio(catalog, config, "queries/it's.wav", k=TOTAL_SEGMENTS + 1)
    finally:
        catalog.close()
