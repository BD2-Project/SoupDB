"""Análisis completo por la API REST (issue #182), de extremo a extremo.

WAV multipart → TestClient → staging único → search_audio (HNSW real) →
build_temporal_evidence → AnalysisResult. Solo se envuelve ``search_audio``
(delegando en la real) para registrar qué ruta de staging recibió.
"""

import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from applications.copyright_detection import analysis as analysis_module
from applications.copyright_detection.analysis import (
    DEFAULT_SEARCH_K,
    CopyrightAnalyzer,
    display_filename,
)
from applications.copyright_detection.app import create_app
from applications.copyright_detection.catalog import AUDIO_SEGMENTS, CopyrightCatalog
from applications.copyright_detection.errors import InvalidAudioError
from applications.copyright_detection.evidence import CopyrightEvidenceError
from applications.copyright_detection.indexing import (
    INDEX_NAME,
    CopyrightIndexConfig,
    ensure_index,
    open_catalog,
    train_codebook,
)
from applications.copyright_detection.ingestion import SongMetadata, ingest_song
from engine.common.catalog import Catalog
from engine.query import execute_sql
from tests.unit.applications.test_copyright_search import write_track

STAGED = re.compile(r"^queries/([0-9a-f]{32})/candidate\.wav$")


@dataclass(frozen=True)
class Built:
    base: Path
    root: Path
    db: Path
    codebook: Path
    cache: Path
    audio: dict[str, bytes]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Built:
    """Dos originales (13 s y 11 s), codebook K=128 y los WAV de los uploads."""
    base = tmp_path_factory.mktemp("copyright_analysis")
    root, db_path = base / "dataset", base / "db"
    write_track(root / "tracks/a.wav", 13.0, seed=1)
    write_track(root / "tracks/b.wav", 11.0, seed=2)
    catalog = Catalog(db_path)
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        ingest_song(db, root, SongMetadata(1, "Alpha", "Ana", "tracks/a.wav"), 1)
        ingest_song(db, root, SongMetadata(2, "Beta", "Bruno", "tracks/b.wav"), 101)
        config = CopyrightIndexConfig(root, base / "codebook.npz", base / "cache", base / "v")
        train_codebook(db, config, base / "train.jsonl")
    finally:
        catalog.close()

    uploads = base / "uploads"  # fuera del dataset: el cliente tiene sus propios archivos
    write_track(uploads / "a.wav", 13.0, seed=1)  # idéntico a tracks/a.wav
    write_track(uploads / "b.wav", 11.0, seed=2)  # idéntico a tracks/b.wav
    write_track(uploads / "short.wav", 2.0, seed=9)
    audio = {name: (uploads / f"{name}.wav").read_bytes() for name in ("a", "b", "short")}
    return Built(base, root, db_path, base / "codebook.npz", base / "cache", audio)


@pytest.fixture
def config(built, tmp_path) -> CopyrightIndexConfig:
    shutil.copytree(built.db, tmp_path / "db")
    return CopyrightIndexConfig(built.root, built.codebook, built.cache, tmp_path / "vectors")


@pytest.fixture
def catalog(config, tmp_path):
    catalog = open_catalog(tmp_path / "db", config)
    try:
        ensure_index(catalog, config)
        yield catalog
    finally:
        catalog.close()


@pytest.fixture
def analyzer(catalog, config) -> CopyrightAnalyzer:
    return CopyrightAnalyzer(catalog, config, search_k=DEFAULT_SEARCH_K)


@pytest.fixture
def client(analyzer) -> TestClient:
    return TestClient(create_app(analyzer))


@pytest.fixture
def staged_paths(monkeypatch) -> list[str]:
    """Rutas que recibe search_audio (se delega en la función real)."""
    paths: list[str] = []
    real = analysis_module.search_audio

    def recording(catalog, config, audio_path, k):
        paths.append(audio_path)
        assert (config.dataset_root / audio_path).is_file()  # existe durante la búsqueda
        return real(catalog, config, audio_path, k=k)

    monkeypatch.setattr(analysis_module, "search_audio", recording)
    return paths


@pytest.fixture(autouse=True)
def staging_is_left_empty(built):
    yield
    staging = built.root / "queries"
    assert not staging.exists() or list(staging.iterdir()) == []


def post(client, content: bytes, name: str = "query.wav"):
    return client.post("/api/v1/analyze", files={"file": (name, content, "audio/wav")})


def best_run(body: dict, song_id: int) -> dict:
    (run,) = [m for m in body["matches"] if m["song_id"] == song_id and m["is_best_run"]]
    return run


# --- E2E ----------------------------------------------------------------------


def test_identical_upload_returns_a_coherent_run_of_that_song(client, built, staged_paths):
    response = post(client, built.audio["a"])

    assert response.status_code == 200
    body = response.json()
    assert re.fullmatch(r"[0-9a-f]{32}", body["analysis_id"])
    assert (body["filename"], body["segment_seconds"]) == ("query.wav", 5.0)
    run = best_run(body, 1)
    assert (run["title"], run["artist"]) == ("Alpha", "Ana")
    assert (run["query_start"], run["query_end"]) == (0.0, 13.0)
    assert (run["source_start"], run["source_end"]) == (0.0, 13.0)
    assert run["segment_count"] == 3
    assert run["score"] == pytest.approx(1.0, abs=1e-6)
    assert run["min_score"] <= run["score"] + 1e-9 and run["score"] <= run["max_score"] + 1e-9
    assert run["alignment_offset_seconds"] == 0.0
    assert body["matches"][0] == run  # la canción con más evidencia va primero
    # Exactamente una best run por canción presente.
    for song_id in {m["song_id"] for m in body["matches"]}:
        assert sum(m["is_best_run"] for m in body["matches"] if m["song_id"] == song_id) == 1
    # El analysis_id es el de la ruta de staging.
    (path,) = staged_paths
    assert STAGED.fullmatch(path).group(1) == body["analysis_id"]


def test_same_filename_with_different_audio_uses_fresh_paths(client, built, staged_paths):
    first = post(client, built.audio["a"], "query.wav").json()
    second = post(client, built.audio["b"], "query.wav").json()

    assert first["analysis_id"] != second["analysis_id"]
    assert len(set(staged_paths)) == 2
    assert all(STAGED.fullmatch(path) for path in staged_paths)
    # Cada respuesta corresponde a su audio: no se reutilizan MFCC del anterior.
    assert first["matches"][0]["song_id"] == 1
    assert second["matches"][0]["song_id"] == 2
    b_run = best_run(second, 2)
    assert (b_run["segment_count"], b_run["query_end"], b_run["source_end"]) == (2, 10.0, 10.0)
    assert b_run["score"] == pytest.approx(1.0, abs=1e-6)


def test_concurrent_analyses_do_not_collide(analyzer, built):
    def run(name):
        from io import BytesIO

        return analyzer.analyze(BytesIO(built.audio[name]), "query.wav")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["a", "b", "a", "b"]))

    assert len({r.analysis_id for r in results}) == 4
    assert [r.matches[0].song_id for r in results] == [1, 2, 1, 2]


# --- cleanup ------------------------------------------------------------------


def test_successful_analysis_removes_its_staging(client, built, staged_paths):
    assert post(client, built.audio["a"]).status_code == 200

    (path,) = staged_paths
    staged = built.root / path
    assert not staged.exists()
    assert not staged.parent.exists()
    assert (built.root / "queries").is_dir()  # la raíz de staging compartida se conserva


@pytest.mark.parametrize(
    ("content_key", "content"),
    [("short", None), ("garbage", b"esto no es audio" * 64), ("empty", b"")],
)
def test_invalid_audio_is_422_and_staging_is_removed(client, built, content_key, content):
    payload = built.audio[content_key] if content is None else content

    response = post(client, payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert str(built.root) not in detail and "candidate" not in detail  # sin rutas internas


def test_staging_is_removed_when_evidence_fails(analyzer, built, monkeypatch):
    def failing(_result):
        raise CopyrightEvidenceError("inconsistente")

    monkeypatch.setattr(analysis_module, "build_temporal_evidence", failing)
    client = TestClient(create_app(analyzer), raise_server_exceptions=False)

    response = post(client, built.audio["a"])

    assert response.status_code == 500
    assert response.json() == {"detail": "Error interno del análisis."}


# --- seguridad del filename ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("../../evil.wav", "evil.wav"),
        ("..\\..\\evil.wav", "evil.wav"),
        ("C:\\x\\evil.wav", "evil.wav"),
        ("/etc/evil.wav", "evil.wav"),
        ("..", "upload"),
        ("", "upload"),
        (None, "upload"),
        ("bad\x00name.wav", "badname.wav"),
    ],
)
def test_display_filename_strips_paths(raw, shown):
    assert display_filename(raw) == shown


def test_traversal_filename_never_decides_where_the_file_is_written(client, built, staged_paths):
    response = post(client, built.audio["a"], "../../evil.wav")

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.wav"
    (path,) = staged_paths
    assert STAGED.fullmatch(path)  # ruta generada en servidor
    assert not list(built.base.rglob("evil.wav"))


# --- infraestructura ------------------------------------------------------------


def test_missing_index_is_503(config, tmp_path, built):
    catalog = open_catalog(tmp_path / "db", config)  # sin ensure_index
    try:
        client = TestClient(create_app(CopyrightAnalyzer(catalog, config)))
        response = post(client, built.audio["a"])
    finally:
        catalog.close()

    assert response.status_code == 503
    assert response.json() == {"detail": "el índice de copyright no está disponible"}


@pytest.mark.parametrize("k", [0, -1, True, 2.0])
def test_search_k_is_validated(catalog, config, k):
    with pytest.raises(ValueError, match="search_k"):
        CopyrightAnalyzer(catalog, config, search_k=k)


@pytest.mark.parametrize("staging", ["", "/abs", "../out", "a/../b", "a\\b", "a#b", "./q"])
def test_staging_dir_is_validated(catalog, config, staging):
    with pytest.raises(ValueError, match="staging_dir"):
        CopyrightAnalyzer(catalog, config, staging_dir=staging)


def test_empty_upload_is_rejected_before_searching(analyzer, staged_paths):
    from io import BytesIO

    with pytest.raises(InvalidAudioError, match="vacío"):
        analyzer.analyze(BytesIO(b""), "q.wav")
    assert staged_paths == []


# --- catálogo inmutable -----------------------------------------------------------


def test_analysis_does_not_modify_catalog_or_index(client, catalog, built):
    db = CopyrightCatalog(catalog)
    binding = catalog.ranked_indexes_for(AUDIO_SEGMENTS)[INDEX_NAME]
    songs_before = db.list_songs()
    rows_before = execute_sql("SELECT * FROM audio_segments", catalog).rows
    vectors_before = len(binding.index)

    assert post(client, built.audio["a"]).status_code == 200
    assert post(client, built.audio["short"]).status_code == 422

    assert db.list_songs() == songs_before
    assert execute_sql("SELECT * FROM audio_segments", catalog).rows == rows_before
    assert len(binding.index) == vectors_before == 5
