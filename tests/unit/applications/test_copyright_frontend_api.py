"""API para el frontend: catálogo, audio de referencia, historial y exportación.

Base real: la demo sintética (ingesta, codebook K=128 y HNSW) se prepara una vez
por módulo; cada test arranca el runtime sobre una copia propia de la base y del
HNSW, porque el historial escribe en la base.
"""

import csv
import io
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from applications.copyright_detection import demo, runtime
from applications.copyright_detection.app import create_app
from applications.copyright_detection.catalog import CopyrightCatalog, Song
from applications.copyright_detection.export import CSV_COLUMNS
from applications.copyright_detection.history import (
    ANALYSES,
    ANALYSIS_MATCHES,
    AnalysisHistory,
)
from applications.copyright_detection.library import CatalogLibrary
from applications.copyright_detection.schemas import AnalysisResult, CopyrightMatch
from applications.copyright_detection.settings import RuntimeSettings
from engine.common.catalog import Catalog
from engine.query import execute_sql
from tests.unit.applications.test_copyright_settings import run_python

API = "/api/v1"
#: Alpha 20 s → 4 segmentos; Beta 15 s → 3; Gamma 12.5 s → 3.
SEGMENTS = {"Alpha": 4, "Beta": 3, "Gamma": 3}
FORBIDDEN_WORDS = ("probab", "detected", "infring", "plagi", "verdict", "dictamen")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> demo.DemoPaths:
    paths, _ = demo.prepare(tmp_path_factory.mktemp("frontend_api") / "demo")
    return paths


@pytest.fixture
def settings(prepared, tmp_path) -> RuntimeSettings:
    shutil.copytree(prepared.db_path, tmp_path / "db")
    shutil.copytree(prepared.vector_storage, tmp_path / "vectors")
    env = prepared.env()
    env.update(
        COPYRIGHT_DB_PATH=str(tmp_path / "db"), COPYRIGHT_VECTOR_STORAGE=str(tmp_path / "vectors")
    )
    return RuntimeSettings.from_env(env)


@pytest.fixture
def client(settings):
    with TestClient(runtime.build_app(settings)) as client:
        yield client


def analyze(client, prepared, filename: str = "sample_query.wav"):
    with prepared.sample_query.open("rb") as audio:
        response = client.post(f"{API}/analyze", files={"file": (filename, audio, "audio/wav")})
    assert response.status_code == 200, response.text
    return response.json()


def history_rows(db_path: Path) -> dict[str, tuple]:
    catalog = Catalog(db_path)
    try:
        return {
            table: execute_sql(f"SELECT * FROM {table}", catalog).rows
            for table in (ANALYSES, ANALYSIS_MATCHES)
        }
    finally:
        catalog.close()


def empty_result(analysis_id: str = "0" * 32) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=analysis_id, filename="silence.wav", segment_seconds=5.0, matches=[]
    )


# --- catálogo ---------------------------------------------------------------------


def test_catalog_lists_the_real_songs_without_internal_paths(client, prepared):
    response = client.get(f"{API}/catalog")

    assert response.status_code == 200
    body = response.json()
    assert body["song_count"] == 3
    assert body["segment_count"] == sum(SEGMENTS.values()) == 10
    assert body["catalog_mutable"] is False
    by_title = {song["title"]: song for song in body["songs"]}
    assert {t: s["segment_count"] for t, s in by_title.items()} == SEGMENTS
    alpha = by_title["Alpha"]
    assert (alpha["song_id"], alpha["artist"]) == (1, "Demo Ensemble")
    assert alpha["duration_seconds"] == pytest.approx(20.0)
    assert alpha["audio_url"] == f"{API}/catalog/1/audio"
    assert "audio_path" not in alpha and "audio_ref" not in response.text
    assert "tracks/" not in response.text
    assert str(prepared.dataset_root) not in response.text


def test_catalog_song_detail_includes_its_segments(client):
    response = client.get(f"{API}/catalog/1")

    assert response.status_code == 200
    body = response.json()
    assert (body["title"], body["segment_count"]) == ("Alpha", 4)
    starts = [(s["segment_index"], s["start_seconds"], s["end_seconds"]) for s in body["segments"]]
    assert starts == [(0, 0.0, 5.0), (1, 5.0, 10.0), (2, 10.0, 15.0), (3, 15.0, 20.0)]
    gamma = client.get(f"{API}/catalog/3").json()
    assert gamma["segments"][-1]["end_seconds"] == pytest.approx(12.5)  # cola de 2.5 s


@pytest.mark.parametrize("song_id", ["999", "4"])
def test_unknown_song_is_404(client, song_id):
    for suffix in ("", "/audio"):
        response = client.get(f"{API}/catalog/{song_id}{suffix}")
        assert response.status_code == 404
        assert "no" in response.json()["detail"]


@pytest.mark.parametrize("song_id", ["0", "-1", "abc", "1.5", str(2**31)])
def test_invalid_song_id_is_rejected(client, song_id):
    for suffix in ("", "/audio"):
        assert client.get(f"{API}/catalog/{song_id}{suffix}").status_code == 422


# --- audio de referencia ----------------------------------------------------------


def test_reference_audio_is_streamed_with_range_support(client, prepared):
    original = (prepared.dataset_root / "tracks" / "alpha.wav").read_bytes()

    full = client.get(f"{API}/catalog/1/audio")
    assert full.status_code == 200
    assert full.content == original
    assert full.headers["content-type"] == "audio/wav"
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-disposition"].startswith("inline")
    assert "alpha.wav" in full.headers["content-disposition"]
    assert str(prepared.dataset_root) not in str(full.headers)

    partial = client.get(f"{API}/catalog/1/audio", headers={"Range": "bytes=100-199"})
    assert partial.status_code == 206
    assert partial.content == original[100:200]
    assert partial.headers["content-range"] == f"bytes 100-199/{len(original)}"

    beyond = client.get(f"{API}/catalog/1/audio", headers={"Range": f"bytes={len(original)}-"})
    assert beyond.status_code == 416


@pytest.mark.parametrize(
    "path",
    [
        "/catalog/..%2F..%2Fdb/audio",
        "/catalog/1%2F..%2F..%2Fdb/audio",
        "/catalog/1/audio/../../db",
        "/catalog/%2E%2E/audio",
        "/catalog/1/audio?path=../../db",
        "/catalog/1/audio?file=C:%5CWindows%5Cwin.ini",
    ],
)
def test_audio_endpoint_never_takes_a_path_from_the_client(client, prepared, path):
    response = client.get(API + path)

    if response.status_code == 200:
        # Los parámetros extra se ignoran: siempre se sirve el archivo de song_id=1.
        assert response.content == (prepared.dataset_root / "tracks" / "alpha.wav").read_bytes()
    else:
        assert response.status_code in (404, 422)


def test_a_stored_path_outside_the_dataset_is_never_served(tmp_path):
    root = tmp_path / "dataset"
    (root / "tracks").mkdir(parents=True)
    (root / "tracks" / "ok.wav").write_bytes(b"RIFF-ok")
    (tmp_path / "secret.wav").write_bytes(b"secret")
    (root / "tracks" / "notes.txt").write_bytes(b"text")
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        # Filas manipuladas: la ruta sale de la raíz, no es canónica o no es audio.
        db.add_song(Song(1, "Escape", "X", "../secret.wav", 1.0))
        db.add_song(Song(2, "Dots", "X", "tracks/../../secret.wav", 1.0))
        db.add_song(Song(3, "Text", "X", "tracks/notes.txt", 1.0))
        db.add_song(Song(4, "Missing", "X", "tracks/missing.wav", 1.0))
        db.add_song(Song(5, "Ok", "X", "tracks/ok.wav", 1.0))
        library = CatalogLibrary(catalog, root)
        client = TestClient(create_app(library=library))

        for song_id in (1, 2, 3, 4):
            assert library.reference_audio(song_id) is None
            response = client.get(f"{API}/catalog/{song_id}/audio")
            assert response.status_code == 404
            assert b"secret" not in response.content
        assert client.get(f"{API}/catalog/5/audio").content == b"RIFF-ok"
    finally:
        catalog.close()


# --- historial --------------------------------------------------------------------


def test_analysis_is_persisted_and_listed(client, prepared):
    result = analyze(client, prepared)

    listing = client.get(f"{API}/analyses")
    assert listing.status_code == 200
    (summary,) = listing.json()
    assert summary["analysis_id"] == result["analysis_id"]
    assert summary["filename"] == "sample_query.wav"
    assert summary["segment_seconds"] == 5.0
    assert summary["match_count"] == len(result["matches"]) > 0
    assert summary["song_count"] == len({m["song_id"] for m in result["matches"]})
    assert summary["best_score"] == max(m["score"] for m in result["matches"])
    assert summary["created_at"].endswith("Z")


def test_analysis_detail_reconstructs_the_full_result(client, prepared):
    result = analyze(client, prepared)

    response = client.get(f"{API}/analyses/{result['analysis_id']}")

    assert response.status_code == 200
    body = response.json()
    created_at = body.pop("created_at")
    assert body == result  # mismas runs, mismo orden, mismos floats
    assert created_at == client.get(f"{API}/analyses").json()[0]["created_at"]
    assert body["matches"][0]["title"] == "Alpha"
    assert not any(word in response.text.lower() for word in FORBIDDEN_WORDS)


def test_history_lists_the_most_recent_first(client, prepared):
    first = analyze(client, prepared, "first.wav")["analysis_id"]
    second = analyze(client, prepared, "second.wav")["analysis_id"]

    listing = client.get(f"{API}/analyses").json()

    assert [a["analysis_id"] for a in listing] == [second, first]
    assert listing[0]["created_at"] >= listing[1]["created_at"]


@pytest.mark.parametrize("analysis_id", ["f" * 32, "abc", "F" * 32, "..%2F..%2Fdb", "' OR 1=1 --"])
def test_unknown_analysis_is_404(client, analysis_id):
    for suffix in ("", "/export?format=json", "/export?format=csv"):
        response = client.get(f"{API}/analyses/{analysis_id}{suffix}")
        assert response.status_code == 404


def test_analysis_without_matches(settings):
    services = runtime.open_analyzer(settings)
    try:
        services.history.save(empty_result())
    finally:
        services.catalog.close()

    with TestClient(runtime.build_app(settings)) as client:
        (summary,) = client.get(f"{API}/analyses").json()
        detail = client.get(f"{API}/analyses/{'0' * 32}").json()
        exported = client.get(f"{API}/analyses/{'0' * 32}/export?format=csv").text

    assert (summary["match_count"], summary["song_count"], summary["best_score"]) == (0, 0, None)
    assert detail["matches"] == []
    assert exported.splitlines() == [",".join(CSV_COLUMNS)]


def test_history_survives_a_restart(settings, prepared):
    with TestClient(runtime.build_app(settings)) as client:
        result = analyze(client, prepared)

    # El primer servidor cerró el Catalog; uno nuevo reabre la base desde disco.
    with TestClient(runtime.build_app(settings)) as client:
        listing = client.get(f"{API}/analyses").json()
        detail = client.get(f"{API}/analyses/{result['analysis_id']}").json()

    assert [a["analysis_id"] for a in listing] == [result["analysis_id"]]
    detail.pop("created_at")
    assert detail == result


def test_upload_is_deleted_and_never_stored(settings, prepared):
    upload = prepared.sample_query.read_bytes()
    with TestClient(runtime.build_app(settings)) as client:
        analyze(client, prepared)
        assert not any(prepared.staging_root.iterdir())  # staging vacío tras el análisis

    # Ningún archivo de la base contiene el audio subido (ni un trozo de su señal).
    probe = upload[len(upload) // 2 : len(upload) // 2 + 256]
    for path in settings.db_path.rglob("*"):
        if path.is_file():
            assert probe not in path.read_bytes(), path
    rows = history_rows(settings.db_path)
    assert len(rows[ANALYSES]) == 1
    assert all(len(row) == 5 for row in rows[ANALYSES])  # sin columna de audio


def test_history_stores_no_absolute_paths(settings, prepared):
    with TestClient(runtime.build_app(settings)) as client:
        result = analyze(client, prepared, filename=r"C:\Users\someone\..\evil.wav")
        unix = analyze(client, prepared, filename="/home/someone/remix.wav")

    assert (result["filename"], unix["filename"]) == ("evil.wav", "remix.wav")
    rows = history_rows(settings.db_path)
    texts = [
        value for table in rows.values() for row in table for value in row if isinstance(value, str)
    ]
    assert texts
    for text in texts:
        assert "/" not in text and "\\" not in text, text
        assert not re.match(r"[A-Za-z]:", text), text
        assert "someone" not in text and str(prepared.dataset_root) not in text


# --- exportación ------------------------------------------------------------------


def test_export_json(client, prepared):
    result = analyze(client, prepared)
    analysis_id = result["analysis_id"]

    response = client.get(f"{API}/analyses/{analysis_id}/export?format=json")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="analysis-{analysis_id}.json"'
    )
    assert json.loads(response.content) == client.get(f"{API}/analyses/{analysis_id}").json()
    # format por defecto: json
    assert client.get(f"{API}/analyses/{analysis_id}/export").content == response.content


def test_export_csv(client, prepared):
    result = analyze(client, prepared, filename="=HYPERLINK(evil).wav")
    analysis_id = result["analysis_id"]

    response = client.get(f"{API}/analyses/{analysis_id}/export?format=csv")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["content-disposition"].endswith(f'analysis-{analysis_id}.csv"')
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == len(result["matches"])
    for order, (row, match) in enumerate(zip(rows, result["matches"], strict=True)):
        assert row["analysis_id"] == analysis_id
        assert row["filename"] == "'=HYPERLINK(evil).wav"  # sin fórmula ejecutable
        assert int(row["match_order"]) == order
        assert int(row["song_id"]) == match["song_id"]
        assert (row["title"], row["artist"]) == (match["title"], match["artist"])
        for key in ("query_start", "query_end", "source_start", "source_end", "score"):
            assert float(row[key]) == match[key]
        assert float(row["duration"]) == pytest.approx(match["query_end"] - match["query_start"])
        assert int(row["segment_count"]) == match["segment_count"]
        assert row["is_best_run"] == str(match["is_best_run"]).lower()


def test_export_rejects_unknown_formats(client, prepared):
    analysis_id = analyze(client, prepared)["analysis_id"]
    for fmt in ("pdf", "xml", "JSON", ""):
        response = client.get(f"{API}/analyses/{analysis_id}/export?format={fmt}")
        assert response.status_code == 422


# --- historial: unidades ----------------------------------------------------------


def test_history_orders_by_creation_time_and_rejects_duplicates(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        CopyrightCatalog(catalog).initialize()
        times = iter(["2026-01-01T00:00:00.000000Z", "2026-03-01T00:00:00.000000Z"])
        history = AnalysisHistory(catalog, clock=lambda: next(times))
        history.initialize()
        history.initialize()  # idempotente
        history.save(empty_result("a" * 32))
        history.save(empty_result("b" * 32))
        assert [s.analysis_id for s in history.list_summaries()] == ["b" * 32, "a" * 32]
        with pytest.raises(Exception, match="ya existe"):
            history.save(empty_result("a" * 32))
        with pytest.raises(Exception, match="inválido"):
            history.save(empty_result("../x"))
    finally:
        catalog.close()


def test_a_failed_save_leaves_no_visible_analysis(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        db.add_song(Song(1, "Alpha", "A", "tracks/a.wav", 20.0))
        history = AnalysisHistory(catalog)
        history.initialize()
        match = CopyrightMatch(
            query_start=0.0,
            query_end=5.0,
            song_id=1,
            title="Alpha",
            artist="A",
            source_start=0.0,
            source_end=5.0,
            score=0.9,
            segment_count=1,
            min_score=0.9,
            max_score=0.9,
            alignment_offset_seconds=0.0,
            is_best_run=True,
        )
        result = AnalysisResult(
            analysis_id="c" * 32, filename="x.wav", segment_seconds=5.0, matches=[match, match]
        )
        real_run = history._run

        def failing(sql):
            if sql.startswith(f"INSERT INTO {ANALYSES} "):
                raise RuntimeError("disco lleno")
            return real_run(sql)

        monkeypatch.setattr(history, "_run", failing)
        with pytest.raises(RuntimeError, match="disco lleno"):
            history.save(result)
        monkeypatch.setattr(history, "_run", real_run)

        assert history.get("c" * 32) is None
        assert history.list_summaries() == []
        assert execute_sql(f"SELECT * FROM {ANALYSIS_MATCHES}", catalog).rows == ()
    finally:
        catalog.close()


# --- configuración del API --------------------------------------------------------


def test_endpoints_without_services_answer_503():
    client = TestClient(create_app())
    for path in (
        "/catalog",
        "/catalog/1",
        "/catalog/1/audio",
        "/analyses",
        f"/analyses/{'a' * 32}",
    ):
        assert client.get(API + path).status_code == 503


def test_cors_allows_range_requests_and_exposes_download_headers():
    client = TestClient(create_app(cors_origins=("http://localhost:5173",)))
    response = client.options(
        f"{API}/catalog/1/audio",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "range",
        },
    )
    assert response.status_code == 200
    assert "range" in response.headers["access-control-allow-headers"].lower()
    simple = client.get(f"{API}/health", headers={"Origin": "http://localhost:5173"})
    exposed = simple.headers["access-control-expose-headers"].lower()
    for header in ("content-disposition", "content-range", "accept-ranges"):
        assert header in exposed


def test_openapi_documents_the_frontend_endpoints_without_verdicts(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for path in (
        "/catalog",
        "/catalog/{song_id}",
        "/catalog/{song_id}/audio",
        "/analyses",
        "/analyses/{analysis_id}",
        "/analyses/{analysis_id}/export",
    ):
        assert set(paths[API + path]) == {"get"}  # sin endpoints de escritura
    assert not any(method != "get" for p, ops in paths.items() if "/catalog" in p for method in ops)
    text = json.dumps(spec["components"]["schemas"]).lower()
    assert "threshold" not in text and "umbral" not in text and "verdict" not in text


def test_importing_the_app_does_not_load_history_or_library():
    output = run_python(
        "import sys, applications.copyright_detection.app; "
        "print(sorted(m for m in ('applications.copyright_detection.history', "
        "'applications.copyright_detection.library', 'engine.common.catalog') "
        "if m in sys.modules))"
    )
    assert output == "[]"
