"""Demo autocontenida y ciclo de vida del servidor real (issue #184).

La demo se prepara una vez por módulo en ``tmp_path`` con las APIs reales
(ingesta, codebook K=128, HNSW). Sobre ella se prueba el runtime con el
``lifespan`` de FastAPI vía ``TestClient``: sin sockets ni esperas.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from soupdb_vector.multimedia.codebook import load_codebook

from applications.copyright_detection import demo, indexing, runtime
from applications.copyright_detection.analysis import CopyrightAnalyzer
from applications.copyright_detection.catalog import AUDIO_SEGMENTS, CopyrightCatalog
from applications.copyright_detection.indexing import (
    CODEBOOK_K,
    INDEX_NAME,
    CopyrightIndexError,
    open_catalog,
    require_index,
)
from applications.copyright_detection.settings import RuntimeSettings
from engine.common.catalog import Catalog

#: Alpha 20 s → 4 segmentos; Beta 15 s → 3; Gamma 12.5 s → 0–5, 5–10, 10–12.5.
EXPECTED_SEGMENTS = {"Alpha": 4, "Beta": 3, "Gamma": 3}
FORBIDDEN_WORDS = ("probab", 'copyright": true', "detected", "infring", "plagi")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> demo.DemoPaths:
    paths, reused = demo.prepare(tmp_path_factory.mktemp("copyright_demo") / "demo")
    assert reused is False
    return paths


@pytest.fixture
def settings(prepared) -> RuntimeSettings:
    return RuntimeSettings.from_env(prepared.env())


@pytest.fixture
def isolated(prepared, tmp_path) -> RuntimeSettings:
    """Copia propia de la base y del HNSW, para poder borrarlas al terminar."""
    shutil.copytree(prepared.db_path, tmp_path / "db")
    shutil.copytree(prepared.vector_storage, tmp_path / "vectors")
    env = prepared.env()
    env.update(COPYRIGHT_DB_PATH=str(tmp_path / "db"))
    env.update(COPYRIGHT_VECTOR_STORAGE=str(tmp_path / "vectors"))
    return RuntimeSettings.from_env(env)


@pytest.fixture
def close_calls(monkeypatch) -> list[Catalog]:
    """Registra cada ``Catalog.close()`` de los catálogos que abre el runtime."""
    calls: list[Catalog] = []
    real_open = indexing.open_catalog

    def tracking_open(db_path, config):
        catalog = real_open(db_path, config)
        real_close = catalog.close

        def close():
            calls.append(catalog)
            real_close()

        catalog.close = close
        return catalog

    monkeypatch.setattr(indexing, "open_catalog", tracking_open)
    return calls


def analyze_sample(client, prepared):
    with prepared.sample_query.open("rb") as audio:
        return client.post(
            "/api/v1/analyze", files={"file": ("sample_query.wav", audio, "audio/wav")}
        )


def snapshot(folder: Path) -> dict[str, str]:
    """Contenido de cada archivo. No se usa mtime: ``Catalog.close()`` reescribe las
    páginas del HNSW con los mismos bytes al hacer flush."""
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# --- demo ---------------------------------------------------------------------


def test_demo_layout(prepared):
    root = prepared.root
    for song in demo.SONGS:
        assert (prepared.dataset_root / "tracks" / song.filename).is_file()
    assert prepared.staging_root.is_dir() and not any(prepared.staging_root.iterdir())
    assert prepared.db_path.is_dir()
    assert prepared.codebook_path.is_file()
    assert prepared.manifest_path.is_file()
    assert prepared.vector_storage.is_dir() and any(prepared.vector_storage.rglob("*"))
    assert prepared.sample_query.parent == root  # fuera del dataset
    assert json.loads(prepared.marker.read_text(encoding="utf-8"))["version"] == 1


def test_demo_songs_and_segments(prepared):
    catalog = Catalog(prepared.db_path)
    try:
        db = CopyrightCatalog(catalog)
        songs = db.list_songs()
        counts = {song.title: len(db.list_segments(song.song_id)) for song in songs}
        paths = {song.audio_path for song in songs}
    finally:
        catalog.close()

    assert counts == EXPECTED_SEGMENTS
    assert paths == {f"tracks/{song.filename}" for song in demo.SONGS}
    assert "sample_query.wav" not in " ".join(paths)  # el candidato no se registra


def test_demo_codebook_has_128_words(prepared):
    centroids, idf = load_codebook(prepared.codebook_path)

    assert centroids.shape[0] == CODEBOOK_K == 128
    assert idf.shape == (128,)


def test_demo_hnsw_reopens_without_rebuilding(prepared, monkeypatch):
    def no_rebuild(*_args, **_kwargs):
        raise AssertionError("el HNSW no debe reconstruirse al reabrir")

    monkeypatch.setattr(indexing, "ensure_index", no_rebuild)
    before = snapshot(prepared.vector_storage)
    catalog = open_catalog(prepared.db_path, prepared.index_config())
    try:
        require_index(catalog)
        binding = catalog.ranked_indexes_for(AUDIO_SEGMENTS)[INDEX_NAME]
        assert len(binding.index) == sum(EXPECTED_SEGMENTS.values())
    finally:
        catalog.close()
    assert snapshot(prepared.vector_storage) == before


def test_demo_audio_is_deterministic_and_the_query_equals_alpha(prepared, tmp_path):
    alpha = prepared.dataset_root / "tracks" / "alpha.wav"
    demo.write_song(demo.SONGS[0], tmp_path / "again.wav")

    assert prepared.sample_query.read_bytes() == alpha.read_bytes()
    assert (tmp_path / "again.wav").read_bytes() == alpha.read_bytes()
    beta = (prepared.dataset_root / "tracks" / "beta.wav").read_bytes()
    assert beta != alpha.read_bytes()


def test_prepare_reuses_a_complete_demo(prepared):
    before = snapshot(prepared.root)

    paths, reused = demo.prepare(prepared.root)

    assert reused is True and paths == prepared
    assert snapshot(prepared.root) == before


def test_prepare_refuses_a_foreign_directory_and_never_deletes(tmp_path):
    root = tmp_path / "busy"
    root.mkdir()
    (root / "notes.txt").write_text("mío", encoding="utf-8")

    with pytest.raises(demo.DemoError, match="no contiene una demo completa"):
        demo.prepare(root)
    assert [p.name for p in root.iterdir()] == ["notes.txt"]


def test_prepare_refuses_a_file_as_root(tmp_path):
    target = tmp_path / "file"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(demo.DemoError, match="no es un directorio"):
        demo.prepare(target)


def test_cli_prints_paths_and_env(prepared, capsys):
    assert demo.main(["prepare", "--root", str(prepared.root)]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Demo reutilizada")
    for line in ("DB: ", "Dataset: ", "Codebook: ", "Vector storage: ", "Query: "):
        assert line in out
    for name, value in prepared.env().items():
        assert f'$env:{name}="{value}"' in out
    assert "python -m applications.copyright_detection.runtime" in out
    assert len(out.splitlines()) < 40


def test_cli_reports_a_foreign_directory(tmp_path, capsys):
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")

    assert demo.main(["prepare", "--root", str(tmp_path)]) == 1
    assert "no contiene una demo completa" in capsys.readouterr().err


# --- runtime: ciclo de vida -------------------------------------------------------


def test_startup_installs_the_analyzer_and_shutdown_closes_once(settings, close_calls):
    app = runtime.build_app(settings)
    assert app.state.analyzer is None  # construir la app no abre nada
    assert close_calls == []

    with TestClient(app) as client:
        analyzer = app.state.analyzer
        assert isinstance(analyzer, CopyrightAnalyzer)
        assert (analyzer.search_k, analyzer.staging_dir) == (5, "queries")
        assert client.get("/api/v1/health").status_code == 200
        assert close_calls == []

    assert app.state.analyzer is None
    assert len(close_calls) == 1


def test_demo_query_through_the_real_runtime(settings, prepared):
    with TestClient(runtime.build_app(settings)) as client:
        response = analyze_sample(client, prepared)

    assert response.status_code == 200
    body = response.json()
    top = body["matches"][0]
    assert (top["title"], top["is_best_run"]) == ("Alpha", True)
    assert (top["query_start"], top["query_end"]) == (0.0, 20.0)
    assert (top["source_start"], top["source_end"]) == (0.0, 20.0)
    assert top["segment_count"] == 4 > 1
    assert top["score"] == pytest.approx(1.0, abs=1e-6)
    # Alpha tiene la run más larga de toda la respuesta y la de mayor score.
    others = [m for m in body["matches"] if m["song_id"] != top["song_id"]]
    assert all(m["segment_count"] < top["segment_count"] for m in others)
    assert all(m["score"] < top["score"] for m in others)
    assert not any(word in response.text.lower() for word in FORBIDDEN_WORDS)
    assert not any(prepared.staging_root.iterdir())


def test_runtime_reopens_the_persistent_hnsw_without_rebuilding(isolated, monkeypatch, prepared):
    def no_rebuild(*_args, **_kwargs):
        raise AssertionError("el runtime no debe crear ni reconstruir el índice")

    monkeypatch.setattr(indexing, "ensure_index", no_rebuild)
    monkeypatch.setattr(indexing, "train_codebook", no_rebuild)
    before = snapshot(isolated.vector_storage)

    with TestClient(runtime.build_app(isolated)) as client:
        assert analyze_sample(client, prepared).status_code == 200

    assert snapshot(isolated.vector_storage) == before


def test_shutdown_releases_every_file(isolated, close_calls):
    with TestClient(runtime.build_app(isolated)) as client:
        assert client.get("/api/v1/health").status_code == 200

    assert len(close_calls) == 1
    # En Windows un handle abierto impediría borrar la base o el índice.
    shutil.rmtree(isolated.db_path)
    shutil.rmtree(isolated.vector_storage)


def test_missing_index_fails_startup_and_closes_the_catalog(
    prepared, tmp_path, close_calls, monkeypatch
):
    catalog = Catalog(tmp_path / "db")  # base de copyright sin HNSW
    try:
        CopyrightCatalog(catalog).initialize()
    finally:
        catalog.close()
    (tmp_path / "vectors").mkdir()
    env = prepared.env()
    env.update(COPYRIGHT_DB_PATH=str(tmp_path / "db"))
    env.update(COPYRIGHT_VECTOR_STORAGE=str(tmp_path / "vectors"))
    created: list[object] = []
    monkeypatch.setattr(runtime, "CopyrightAnalyzer", lambda *a, **k: created.append(a))
    app = runtime.build_app(RuntimeSettings.from_env(env))

    with pytest.raises(CopyrightIndexError, match=INDEX_NAME):
        with TestClient(app):
            pass

    assert created == []  # sin índice no se crea el analizador
    assert app.state.analyzer is None
    assert len(close_calls) == 1
    shutil.rmtree(tmp_path / "db")  # ningún handle quedó abierto


def test_failure_after_opening_closes_the_catalog(settings, close_calls, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("fallo al crear el analizador")

    monkeypatch.setattr(runtime, "CopyrightAnalyzer", broken)

    with pytest.raises(RuntimeError, match="fallo al crear"):
        runtime.open_analyzer(settings)

    assert len(close_calls) == 1


@pytest.mark.parametrize("variable", ["COPYRIGHT_DB_PATH", "COPYRIGHT_VECTOR_STORAGE"])
def test_missing_directories_fail_before_opening_anything(
    prepared, tmp_path, close_calls, variable
):
    env = prepared.env()
    env[variable] = str(tmp_path / "nope")

    with pytest.raises(CopyrightIndexError, match=variable):
        runtime.open_analyzer(RuntimeSettings.from_env(env))

    assert close_calls == []
    assert not (tmp_path / "nope").exists()  # no crea una base vacía


def test_runtime_app_attribute_reads_the_environment(prepared, monkeypatch):
    for name in list(prepared.env()):
        monkeypatch.delenv(name, raising=False)
    runtime._app_from_env.cache_clear()
    try:
        with pytest.raises(ValueError, match="COPYRIGHT_DB_PATH"):
            _ = runtime.app
        for name, value in prepared.env().items():
            monkeypatch.setenv(name, value)
        runtime._app_from_env.cache_clear()
        assert runtime.app is runtime.app  # se construye una vez
        assert runtime.app.state.analyzer is None  # aún sin lifespan
    finally:
        runtime._app_from_env.cache_clear()


def test_main_runs_uvicorn_with_host_and_port(prepared, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    for name, value in prepared.env().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("COPYRIGHT_HOST", "0.0.0.0")
    monkeypatch.setenv("COPYRIGHT_PORT", "8123")

    runtime.main()

    ((app, kwargs),) = calls
    assert kwargs == {"host": "0.0.0.0", "port": 8123}
    assert app.state.analyzer is None  # el lifespan lo instalará uvicorn al arrancar


def test_main_reports_invalid_configuration(monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("no debe arrancar"))
    for name in ("COPYRIGHT_DB_PATH", "COPYRIGHT_SEARCH_K"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit, match="COPYRIGHT_DB_PATH"):
        runtime.main()
