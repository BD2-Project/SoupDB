"""Configuración del servidor por entorno, CORS e import ligero (issue #184).

Sin base de datos: solo parsing de ``COPYRIGHT_*`` y la app HTTP. El ciclo de
vida real (abrir/cerrar el ``Catalog``) se prueba en ``test_copyright_demo.py``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from applications.copyright_detection.app import create_app
from applications.copyright_detection.settings import (
    CODEBOOK_PATH,
    CORS_ORIGINS,
    DB_PATH,
    HOST,
    PORT,
    REQUIRED_PATHS,
    SEARCH_K,
    STAGING_DIR,
    RuntimeConfigError,
    RuntimeSettings,
    parse_cors_origins,
)


@pytest.fixture
def env(tmp_path) -> dict[str, str]:
    """Las cinco rutas obligatorias, absolutas."""
    return {
        "COPYRIGHT_DB_PATH": str(tmp_path / "db"),
        "COPYRIGHT_DATASET_ROOT": str(tmp_path / "dataset"),
        "COPYRIGHT_CODEBOOK_PATH": str(tmp_path / "artifacts" / "codebook.npz"),
        "COPYRIGHT_DESCRIPTOR_CACHE": str(tmp_path / "artifacts" / "cache"),
        "COPYRIGHT_VECTOR_STORAGE": str(tmp_path / "artifacts" / "vectors"),
    }


# --- rutas obligatorias ---------------------------------------------------------


def test_required_paths_and_defaults(env, tmp_path):
    settings = RuntimeSettings.from_env(env)

    assert settings.db_path == tmp_path / "db"
    assert settings.dataset_root == tmp_path / "dataset"
    assert settings.codebook_path == tmp_path / "artifacts" / "codebook.npz"
    assert settings.descriptor_cache == tmp_path / "artifacts" / "cache"
    assert settings.vector_storage == tmp_path / "artifacts" / "vectors"
    assert all(isinstance(p, Path) for p in (settings.db_path, settings.vector_storage))
    assert (settings.search_k, settings.staging_dir) == (5, "queries")
    assert (settings.host, settings.port, settings.cors_origins) == ("127.0.0.1", 8000, ())


@pytest.mark.parametrize("name", REQUIRED_PATHS)
@pytest.mark.parametrize("value", [None, "", "   "])
def test_each_required_path_is_mandatory(env, name, value):
    if value is None:
        del env[name]
    else:
        env[name] = value

    with pytest.raises(RuntimeConfigError, match=name):
        RuntimeSettings.from_env(env)


def test_all_missing_paths_are_listed_together():
    with pytest.raises(RuntimeConfigError) as info:
        RuntimeSettings.from_env({})

    assert all(name in str(info.value) for name in REQUIRED_PATHS)


def test_relative_paths_are_resolved_against_the_cwd(env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env[DB_PATH] = "demo/db"

    settings = RuntimeSettings.from_env(env)

    assert settings.db_path.is_absolute()
    assert settings.db_path == tmp_path / "demo" / "db"


def test_codebook_must_be_npz(env):
    env[CODEBOOK_PATH] = env[CODEBOOK_PATH].removesuffix(".npz") + ".bin"

    with pytest.raises(RuntimeConfigError, match=CODEBOOK_PATH):
        RuntimeSettings.from_env(env)


def test_reads_os_environ_by_default(env, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(SEARCH_K, "9")

    assert RuntimeSettings.from_env().search_k == 9


# --- enteros -------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("7", 7), (" 12 ", 12), ("1000", 1000)])
def test_search_k_valid(env, raw, expected):
    env[SEARCH_K] = raw

    assert RuntimeSettings.from_env(env).search_k == expected


@pytest.mark.parametrize("raw", ["0", "-1", "true", "True", "5.0", "abc", "0x5", "1001", "+5"])
def test_search_k_invalid_is_an_error_not_a_default(env, raw):
    env[SEARCH_K] = raw

    with pytest.raises(RuntimeConfigError, match=SEARCH_K):
        RuntimeSettings.from_env(env)


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("8080", 8080), ("65535", 65535)])
def test_port_valid(env, raw, expected):
    env[PORT] = raw

    assert RuntimeSettings.from_env(env).port == expected


@pytest.mark.parametrize("raw", ["0", "65536", "-80", "http", "8000.0", "false"])
def test_port_invalid(env, raw):
    env[PORT] = raw

    with pytest.raises(RuntimeConfigError, match=PORT):
        RuntimeSettings.from_env(env)


@pytest.mark.parametrize("name", [SEARCH_K, PORT, STAGING_DIR, HOST])
def test_empty_optional_uses_default(env, name):
    env[name] = ""

    settings = RuntimeSettings.from_env(env)

    assert (settings.search_k, settings.port) == (5, 8000)
    assert (settings.staging_dir, settings.host) == ("queries", "127.0.0.1")


def test_host_and_staging_dir_are_configurable(env):
    env[HOST] = "0.0.0.0"
    env[STAGING_DIR] = "uploads/tmp"

    settings = RuntimeSettings.from_env(env)

    assert (settings.host, settings.staging_dir) == ("0.0.0.0", "uploads/tmp")


@pytest.mark.parametrize("raw", ["/abs", "../out", "a\\b", "a#b", "./q", "a/../b"])
def test_staging_dir_invalid(env, raw):
    env[STAGING_DIR] = raw

    with pytest.raises(RuntimeConfigError, match=STAGING_DIR):
        RuntimeSettings.from_env(env)


# --- CORS ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("  ,  , ", ()),
        ("http://localhost:5173", ("http://localhost:5173",)),
        (
            " http://localhost:5173 , http://127.0.0.1:5173/ ,,https://demo.example ",
            ("http://localhost:5173", "http://127.0.0.1:5173", "https://demo.example"),
        ),
        ("http://a:1,http://a:1", ("http://a:1",)),
    ],
)
def test_cors_origins_are_normalized(raw, expected):
    assert parse_cors_origins(raw) == expected


@pytest.mark.parametrize(
    "raw", ["*", "localhost:5173", "ftp://x", "http://x/app", "http://x?q=1", "http://"]
)
def test_cors_origins_invalid(raw):
    with pytest.raises(RuntimeConfigError, match=CORS_ORIGINS):
        parse_cors_origins(raw)


def test_cors_from_env(env):
    env[CORS_ORIGINS] = "http://localhost:5173, http://127.0.0.1:5173"

    origins = RuntimeSettings.from_env(env).cors_origins

    assert origins == ("http://localhost:5173", "http://127.0.0.1:5173")


def preflight(client, origin):
    return client.options(
        "/api/v1/analyze",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


def test_without_origins_there_is_no_cors():
    client = TestClient(create_app())

    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})

    assert "access-control-allow-origin" not in response.headers
    assert preflight(client, "http://localhost:5173").status_code == 405


def test_configured_origins_allow_multipart_post_from_the_frontend():
    client = TestClient(create_app(cors_origins=("http://localhost:5173", "http://127.0.0.1:5173")))

    response = preflight(client, "http://127.0.0.1:5173")

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
    simple = client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
    assert simple.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unlisted_origins_and_is_never_universal():
    client = TestClient(create_app(cors_origins=("http://localhost:5173",)))

    response = preflight(client, "http://evil.example")

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
    simple = client.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


# --- import ligero ------------------------------------------------------------


def run_python(code: str) -> str:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("COPYRIGHT_")}
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=clean
    ).stdout.strip()


def test_importing_the_app_loads_neither_runtime_nor_pipeline():
    output = run_python(
        "import sys, applications.copyright_detection.app; "
        "print(sorted(m for m in ('librosa', 'soupdb_vector', "
        "'applications.copyright_detection.analysis', "
        "'applications.copyright_detection.indexing', "
        "'applications.copyright_detection.runtime') if m in sys.modules))"
    )

    assert output == "[]"


def test_importing_the_runtime_needs_no_configuration():
    # Sin COPYRIGHT_*: importar no lee el entorno ni construye la app.
    output = run_python(
        "import applications.copyright_detection.runtime as r; print('app' in vars(r))"
    )

    assert output == "False"
