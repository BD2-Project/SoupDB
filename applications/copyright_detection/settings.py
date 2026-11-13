"""Configuración del servidor de copyright leída de variables de entorno.

Solo parsea y valida: no abre la base, no comprueba que los archivos existan y
no importa el pipeline multimedia (stdlib pura). La apertura real vive en
:mod:`applications.copyright_detection.runtime`.

Variables (las cinco rutas son obligatorias):

==============================  ===============  =====================================
Variable                        Por defecto      Qué es
==============================  ===============  =====================================
``COPYRIGHT_DB_PATH``           obligatoria      Directorio de la base SoupDB
``COPYRIGHT_DATASET_ROOT``      obligatoria      Raíz del dataset (``audio_path`` relativo)
``COPYRIGHT_CODEBOOK_PATH``     obligatoria      ``codebook.npz`` K=128 ya entrenado
``COPYRIGHT_DESCRIPTOR_CACHE``  obligatoria      Caché de MFCC del perfil
``COPYRIGHT_VECTOR_STORAGE``    obligatoria      Carpeta del HNSW persistente
``COPYRIGHT_SEARCH_K``          ``5``            Vecinos HNSW por segmento (> 0)
``COPYRIGHT_STAGING_DIR``       ``queries``      Staging de uploads, relativo al dataset
``COPYRIGHT_HOST``              ``127.0.0.1``    Interfaz de escucha
``COPYRIGHT_PORT``              ``8000``         Puerto (1–65535)
``COPYRIGHT_CORS_ORIGINS``      vacío (sin CORS) Orígenes permitidos, separados por comas
==============================  ===============  =====================================

Las rutas relativas se resuelven contra el directorio de trabajo del proceso
al leer la configuración. Una variable opcional ausente o vacía usa su valor
por defecto; un valor presente pero inválido es un error, nunca un default.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

DB_PATH = "COPYRIGHT_DB_PATH"
DATASET_ROOT = "COPYRIGHT_DATASET_ROOT"
CODEBOOK_PATH = "COPYRIGHT_CODEBOOK_PATH"
DESCRIPTOR_CACHE = "COPYRIGHT_DESCRIPTOR_CACHE"
VECTOR_STORAGE = "COPYRIGHT_VECTOR_STORAGE"
SEARCH_K = "COPYRIGHT_SEARCH_K"
STAGING_DIR = "COPYRIGHT_STAGING_DIR"
HOST = "COPYRIGHT_HOST"
PORT = "COPYRIGHT_PORT"
CORS_ORIGINS = "COPYRIGHT_CORS_ORIGINS"

REQUIRED_PATHS = (DB_PATH, DATASET_ROOT, CODEBOOK_PATH, DESCRIPTOR_CACHE, VECTOR_STORAGE)

DEFAULT_SEARCH_K = 5
DEFAULT_STAGING_DIR = "queries"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
#: Tope de vecinos por segmento: evita consultas absurdamente grandes por error.
MAX_SEARCH_K = 1000

_DIGITS = re.compile(r"[0-9]+")


class RuntimeConfigError(ValueError):
    """Una variable de entorno falta o tiene un valor inválido."""


def validate_staging_dir(staging_dir: str) -> str:
    """Devuelve ``staging_dir`` si es relativa, canónica y con ``/``; si no, ``ValueError``."""
    posix = PurePosixPath(staging_dir)
    if (
        not staging_dir
        or posix.is_absolute()
        or PureWindowsPath(staging_dir).drive
        or posix.as_posix() != staging_dir
        or any(part in (".", "..") for part in posix.parts)
        or "\\" in staging_dir
        or "#" in staging_dir
    ):
        raise ValueError(f"staging_dir debe ser una ruta relativa canónica: {staging_dir!r}")
    return staging_dir


def parse_int(name: str, raw: str, *, minimum: int, maximum: int) -> int:
    """Entero decimal en ``[minimum, maximum]``; rechaza ``true``, ``5.0``, ``0x5``, etc."""
    value = raw.strip()
    if not _DIGITS.fullmatch(value):
        raise RuntimeConfigError(f"{name} debe ser un entero, no {raw!r}")
    number = int(value)
    if not minimum <= number <= maximum:
        raise RuntimeConfigError(f"{name} debe estar entre {minimum} y {maximum}, no {number}")
    return number


def parse_cors_origins(raw: str) -> tuple[str, ...]:
    """``"http://a:5173, http://b"`` → ``("http://a:5173", "http://b")``; vacío → ``()``.

    Cada origen debe ser ``http(s)://host[:puerto]`` sin ruta. ``*`` se rechaza:
    los orígenes se enumeran explícitamente.
    """
    origins: list[str] = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise RuntimeConfigError(f"{CORS_ORIGINS} no admite '*'; enumera los orígenes")
        parts = urlsplit(origin)
        if (
            parts.scheme not in ("http", "https")
            or not parts.netloc
            or parts.path
            or parts.query
            or parts.fragment
        ):
            raise RuntimeConfigError(
                f"{CORS_ORIGINS}: origen inválido {item.strip()!r} (usa http(s)://host[:puerto])"
            )
        if origin not in origins:
            origins.append(origin)
    return tuple(origins)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Configuración validada del servidor de copyright."""

    db_path: Path
    dataset_root: Path
    codebook_path: Path
    descriptor_cache: Path
    vector_storage: Path
    search_k: int = DEFAULT_SEARCH_K
    staging_dir: str = DEFAULT_STAGING_DIR
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeSettings":
        """Lee la configuración de ``env`` (por defecto ``os.environ``)."""
        source = os.environ if env is None else env

        def optional(name: str) -> str | None:
            value = source.get(name)
            return None if value is None or not value.strip() else value

        missing = [name for name in REQUIRED_PATHS if optional(name) is None]
        if missing:
            raise RuntimeConfigError(
                "faltan variables de entorno obligatorias: " + ", ".join(missing)
            )
        paths = {
            name: Path(source[name].strip()).expanduser().absolute() for name in REQUIRED_PATHS
        }
        if paths[CODEBOOK_PATH].suffix != ".npz":
            raise RuntimeConfigError(
                f"{CODEBOOK_PATH} debe terminar en .npz: {paths[CODEBOOK_PATH]}"
            )

        raw_k = optional(SEARCH_K)
        raw_port = optional(PORT)
        raw_staging = optional(STAGING_DIR)
        staging = DEFAULT_STAGING_DIR if raw_staging is None else raw_staging.strip()
        try:
            validate_staging_dir(staging)
        except ValueError:
            raise RuntimeConfigError(
                f"{STAGING_DIR} debe ser una ruta relativa canónica con '/', no {raw_staging!r}"
            ) from None

        return cls(
            db_path=paths[DB_PATH],
            dataset_root=paths[DATASET_ROOT],
            codebook_path=paths[CODEBOOK_PATH],
            descriptor_cache=paths[DESCRIPTOR_CACHE],
            vector_storage=paths[VECTOR_STORAGE],
            search_k=(
                DEFAULT_SEARCH_K
                if raw_k is None
                else parse_int(SEARCH_K, raw_k, minimum=1, maximum=MAX_SEARCH_K)
            ),
            staging_dir=staging,
            host=(optional(HOST) or DEFAULT_HOST).strip(),
            port=(
                DEFAULT_PORT
                if raw_port is None
                else parse_int(PORT, raw_port, minimum=1, maximum=65535)
            ),
            cors_origins=parse_cors_origins(source.get(CORS_ORIGINS, "")),
        )
