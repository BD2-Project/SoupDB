"""API REST de copyright: health, /analyze con analizador inyectado, schemas y OpenAPI.

Aquí el analizador es un doble inyectado por ``create_app``; el análisis real
(HNSW, staging, evidencia) se prueba de extremo a extremo en
``test_copyright_analysis.py``.
"""

import subprocess
import sys
from statistics import fmean

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from applications.copyright_detection import app as app_module
from applications.copyright_detection import config
from applications.copyright_detection.app import (
    ANALYZER_NOT_CONFIGURED,
    INTERNAL_ERROR,
    create_app,
)
from applications.copyright_detection.errors import AnalysisUnavailableError, InvalidAudioError
from applications.copyright_detection.schemas import AnalysisResult, CopyrightMatch

MATCH = {
    "query_start": 0.0,
    "query_end": 15.0,
    "song_id": 1,
    "title": "Canción",
    "artist": "Artista",
    "source_start": 35.0,
    "source_end": 50.0,
    "score": 0.94,
    "segment_count": 3,
    "min_score": 0.92,
    "max_score": 0.95,
    "alignment_offset_seconds": 35.0,
    "is_best_run": True,
}

FORBIDDEN_WORDS = ("probability", "probabilidad", "is_copyright", "infring", "plagi")


class FakeAnalyzer:
    """Doble del servicio: registra la llamada y devuelve o lanza lo configurado."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[tuple[bytes, str | None]] = []

    def analyze(self, upload, filename):
        self.calls.append((upload.read(), filename))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def post_file(client, name="query.wav", content=b"RIFF-audio"):
    return client.post("/api/v1/analyze", files={"file": (name, content, "audio/wav")})


@pytest.fixture
def client():
    return TestClient(create_app())


# --- health ---------------------------------------------------------------


def test_health_returns_exact_payload(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "soupdb-copyright-api"}


# --- /analyze sin analizador ---------------------------------------------------------


def test_analyze_without_analyzer_is_service_unavailable(client):
    response = post_file(client)

    assert response.status_code == 503
    assert response.json() == {"detail": ANALYZER_NOT_CONFIGURED}


def test_not_implemented_contract_is_gone():
    assert not hasattr(app_module, "ANALYZE_NOT_IMPLEMENTED")


def test_analyze_requires_file_field(client):
    assert client.post("/api/v1/analyze").status_code == 422
    configured = TestClient(create_app(FakeAnalyzer(None)))
    assert configured.post("/api/v1/analyze").status_code == 422


def test_analyze_only_accepts_post(client):
    assert client.get("/api/v1/analyze").status_code == 405


# --- /analyze con analizador inyectado ----------------------------------------------


def test_analyze_returns_the_analyzer_result_as_json():
    result = AnalysisResult(
        analysis_id="abc", filename="query.wav", segment_seconds=5.0, matches=[MATCH]
    )
    analyzer = FakeAnalyzer(result)

    response = post_file(TestClient(create_app(analyzer)), "query.wav", b"bytes-del-audio")

    assert response.status_code == 200
    assert response.json() == result.model_dump(mode="json")
    assert analyzer.calls == [(b"bytes-del-audio", "query.wav")]


def test_analyze_with_no_evidence_is_ok():
    result = AnalysisResult(analysis_id="abc", filename="q.wav", segment_seconds=5.0)

    response = post_file(TestClient(create_app(FakeAnalyzer(result))))

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_invalid_audio_maps_to_422():
    analyzer = FakeAnalyzer(InvalidAudioError("el audio no se puede decodificar"))

    response = post_file(TestClient(create_app(analyzer)))

    assert response.status_code == 422
    assert response.json() == {"detail": "el audio no se puede decodificar"}


def test_unavailable_index_maps_to_503():
    analyzer = FakeAnalyzer(AnalysisUnavailableError("el índice no está disponible"))

    response = post_file(TestClient(create_app(analyzer)))

    assert response.status_code == 503
    assert response.json() == {"detail": "el índice no está disponible"}


def test_unexpected_error_maps_to_500_without_traceback():
    analyzer = FakeAnalyzer(RuntimeError("secreto interno /ruta/privada"))

    response = post_file(TestClient(create_app(analyzer), raise_server_exceptions=False))

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_ERROR}
    assert "secreto" not in response.text and "Traceback" not in response.text


def test_importing_the_api_does_not_load_the_multimedia_pipeline():
    code = (
        "import sys, applications.copyright_detection.app; "
        "print(sorted(m for m in ('librosa', 'soupdb_vector', "
        "'applications.copyright_detection.analysis') if m in sys.modules))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout

    assert output.strip() == "[]"


# --- schemas ----------------------------------------------------------------


def test_match_represents_a_temporal_run():
    match = CopyrightMatch(**MATCH)

    assert (match.query_start, match.query_end) == (0.0, 15.0)
    assert (match.source_start, match.source_end) == (35.0, 50.0)
    assert (match.song_id, match.title, match.artist) == (1, "Canción", "Artista")
    assert (match.segment_count, match.min_score, match.score, match.max_score) == (
        3,
        0.92,
        0.94,
        0.95,
    )
    assert match.is_best_run is True
    assert match.model_dump() == MATCH


def test_analysis_result_contract():
    result = AnalysisResult(
        analysis_id="a-1", filename="q.wav", segment_seconds=config.SEGMENT_SECONDS
    )

    assert result.segment_seconds == 5.0
    assert result.matches == []


@pytest.mark.parametrize(
    "override",
    [
        {"query_start": 10.0, "query_end": 10.0},
        {"source_start": 50.0, "source_end": 35.0},
        {"query_start": -1.0},
        {"segment_count": 0},
        {"segment_count": True},
        {"segment_count": 2.0},
        {"score": -0.1, "min_score": -0.1},
        {"score": 1.5, "max_score": 1.5},
        {"min_score": 0.96},
        {"max_score": 0.93},
        {"min_score": 0.95, "max_score": 0.92},
        {"is_best_run": "true"},
        {"is_best_run": 1},
    ],
)
def test_match_rejects_invalid_values(override):
    with pytest.raises(ValidationError):
        CopyrightMatch(**{**MATCH, **override})


def test_match_accepts_mean_rounding_above_max():
    # fmean([0.1] * 3) == 0.10000000000000002 > 0.1: evidencia real, no se rechaza.
    scores = [0.1, 0.1, 0.1]
    match = CopyrightMatch(
        **{**MATCH, "score": fmean(scores), "min_score": min(scores), "max_score": max(scores)}
    )

    assert match.score > match.max_score


def test_segment_seconds_must_be_positive():
    with pytest.raises(ValidationError):
        AnalysisResult(analysis_id="a", filename="x.wav", segment_seconds=0)


def test_score_is_documented_as_mean_similarity_not_probability():
    description = CopyrightMatch.model_fields["score"].description.lower()

    assert "media de los scores de similitud" in description
    assert "no es una probabilidad" in description


def test_planned_configuration():
    assert config.SEGMENT_SECONDS == 5.0
    assert config.INDEX_KIND == "HNSW"
    assert config.METRIC == "euclidean"


def test_segment_seconds_matches_audio_pipeline():
    # Import local: carga librosa solo para este test, no al colectar el módulo.
    from soupdb_vector.multimedia.audio_features import SEGMENT_SECONDS

    assert config.SEGMENT_SECONDS == SEGMENT_SECONDS


# --- OpenAPI ----------------------------------------------------------------------


@pytest.fixture
def openapi(client):
    return client.get("/openapi.json").json()


def test_openapi_lists_endpoints(openapi):
    assert "get" in openapi["paths"]["/api/v1/health"]
    assert "post" in openapi["paths"]["/api/v1/analyze"]


def test_openapi_documents_analyze_contract(openapi):
    analyze = openapi["paths"]["/api/v1/analyze"]["post"]
    schemas = openapi["components"]["schemas"]

    assert "multipart/form-data" in analyze["requestBody"]["content"]
    ok = analyze["responses"]["200"]["content"]["application/json"]["schema"]
    assert ok == {"$ref": "#/components/schemas/AnalysisResult"}
    assert {"200", "422", "503", "500"} <= set(analyze["responses"])
    assert "501" not in analyze["responses"]

    result_props = schemas["AnalysisResult"]["properties"]
    assert {"analysis_id", "filename", "segment_seconds", "matches"} <= set(result_props)
    match_props = schemas["CopyrightMatch"]["properties"]
    assert set(MATCH) <= set(match_props)
    assert "similitud" in match_props["score"]["description"]


def test_openapi_has_no_probability_or_verdict_fields(openapi):
    for schema in openapi["components"]["schemas"].values():
        for field in schema.get("properties", {}):
            assert not any(word in field.lower() for word in FORBIDDEN_WORDS), field
