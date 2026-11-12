"""Base del API REST de copyright (issue #170): health, contrato de /analyze y OpenAPI."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from applications.copyright_detection import config
from applications.copyright_detection.app import ANALYZE_NOT_IMPLEMENTED, create_app
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
}


@pytest.fixture
def client():
    return TestClient(create_app())


# --- health ---------------------------------------------------------------


def test_health_returns_exact_payload(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "soupdb-copyright-api"}


# --- /analyze ---------------------------------------------------------------


def test_analyze_with_upload_is_explicitly_not_implemented(client):
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("query.mp3", b"ID3\x00fake-audio", "audio/mpeg")},
    )

    assert response.status_code == 501
    assert response.json() == {"detail": ANALYZE_NOT_IMPLEMENTED}


def test_analyze_never_returns_matches_while_not_implemented(client):
    response = client.post("/api/v1/analyze", files={"file": ("query.wav", b"RIFF", "audio/wav")})

    assert "matches" not in response.json()


def test_analyze_requires_file_field(client):
    response = client.post("/api/v1/analyze")

    assert response.status_code == 422


def test_analyze_only_accepts_post(client):
    assert client.get("/api/v1/analyze").status_code == 405


# --- schemas ----------------------------------------------------------------


def test_analysis_result_represents_expected_contract():
    result = AnalysisResult(
        analysis_id="a-1",
        filename="query.mp3",
        segment_seconds=config.SEGMENT_SECONDS,
        matches=[CopyrightMatch(**MATCH)],
    )

    assert result.segment_seconds == 5.0
    match = result.matches[0]
    assert (match.query_start, match.query_end) == (0.0, 15.0)
    assert (match.source_start, match.source_end) == (35.0, 50.0)
    assert (match.song_id, match.title, match.artist) == (1, "Canción", "Artista")
    assert match.score == 0.94
    assert result.model_dump()["matches"][0] == MATCH


def test_analysis_result_allows_no_matches():
    result = AnalysisResult(analysis_id="a-2", filename="x.wav", segment_seconds=5.0)

    assert result.matches == []


@pytest.mark.parametrize(
    "override",
    [
        {"query_start": 10.0, "query_end": 5.0},
        {"source_start": 50.0, "source_end": 35.0},
        {"query_start": -1.0},
        {"score": -0.1},
        {"score": 1.5},
    ],
)
def test_match_rejects_invalid_values(override):
    with pytest.raises(ValidationError):
        CopyrightMatch(**{**MATCH, **override})


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"query_start": 5.0, "query_end": 5.0}, "query_end"),
        ({"source_start": 10.0, "source_end": 10.0}, "source_end"),
    ],
)
def test_match_rejects_zero_duration_intervals(override, field):
    with pytest.raises(ValidationError, match=f"{field} debe ser >"):
        CopyrightMatch(**{**MATCH, **override})


def test_segment_seconds_must_be_positive():
    with pytest.raises(ValidationError):
        AnalysisResult(analysis_id="a", filename="x.wav", segment_seconds=0)


def test_score_is_documented_as_similarity_not_probability():
    description = CopyrightMatch.model_fields["score"].description.lower()

    assert "similitud" in description
    assert "no es una probabilidad" in description


# --- configuración prevista ------------------------------------------------------


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
    assert "501" in analyze["responses"]

    result_props = schemas["AnalysisResult"]["properties"]
    assert {"analysis_id", "filename", "segment_seconds", "matches"} <= set(result_props)
    assert set(MATCH) <= set(schemas["CopyrightMatch"]["properties"])
    assert "similitud" in schemas["CopyrightMatch"]["properties"]["score"]["description"]
