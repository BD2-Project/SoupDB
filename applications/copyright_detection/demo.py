"""Demo autocontenida de la aplicación de copyright: genera, ingesta e indexa.

Uso::

    uv run --group copyright-api python -m applications.copyright_detection.demo prepare \\
        --root .demo/copyright

Genera en runtime tres canciones sintéticas deterministas (Alpha, Beta y Gamma;
sin material externo ni con copyright, y sin WAV versionados) y prepara una base
lista para el servidor con las APIs reales de la aplicación::

    WAV originales → CopyrightCatalog.initialize() → ingest_song()
    → train_codebook(K=128) → open_catalog() → ensure_index(HNSW euclidean) → close()

Estructura creada en ``--root``::

    dataset/tracks/{alpha,beta,gamma}.wav   originales (``audio_path`` relativo a dataset/)
    dataset/queries/                        raíz del staging de uploads (vacía)
    db/                                     base SoupDB (songs, audio_segments)
    artifacts/codebook.npz                  codebook K=128 + IDF
    artifacts/cache/                        caché de MFCC
    artifacts/vectors/                      HNSW persistente
    artifacts/training_manifest.jsonl       manifest de entrenamiento (artefacto)
    sample_query.wav                        audio de prueba para curl (idéntico a Alpha)
    demo.json                               marca de demo completa

``sample_query.wav`` es solo un input de usuario: no se registra en SoupDB.

Idempotencia: si ``--root`` ya contiene una demo completa y válida, se reutiliza
sin tocarla. Si existe con otro contenido (o una preparación a medias), falla y
pide borrarlo a mano: la herramienta nunca borra directorios.
"""

import argparse
import json
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from soupdb_vector.multimedia.audio_features import SAMPLE_RATE

from applications.copyright_detection.catalog import CopyrightCatalog
from applications.copyright_detection.indexing import (
    CopyrightIndexConfig,
    CopyrightIndexError,
    ensure_index,
    open_catalog,
    require_index,
    train_codebook,
)
from applications.copyright_detection.ingestion import SongMetadata, ingest_song
from applications.copyright_detection.settings import (
    CODEBOOK_PATH,
    DATASET_ROOT,
    DB_PATH,
    DEFAULT_STAGING_DIR,
    DESCRIPTOR_CACHE,
    VECTOR_STORAGE,
)
from engine.common.catalog import Catalog

DEFAULT_ROOT = Path(".demo/copyright")
MARKER = "demo.json"
MARKER_VERSION = 1


@dataclass(frozen=True, slots=True)
class DemoSong:
    song_id: int
    title: str
    artist: str
    filename: str
    seconds: float
    #: Frecuencia base (Hz): cada canción ocupa una zona espectral distinta.
    base_hz: float
    seed: int
    first_segment_id: int


#: Duraciones elegidas para cubrir el caso de cola: 12.5 s da 0–5, 5–10 y 10–12.5.
SONGS = (
    DemoSong(1, "Alpha", "Demo Ensemble", "alpha.wav", 20.0, 220.0, 11, 1),
    DemoSong(2, "Beta", "Demo Ensemble", "beta.wav", 15.0, 520.0, 22, 101),
    DemoSong(3, "Gamma", "Demo Ensemble", "gamma.wav", 12.5, 1200.0, 33, 201),
)
QUERY_SOURCE = SONGS[0]


class DemoError(Exception):
    """El directorio de la demo no se puede preparar o reutilizar."""


@dataclass(frozen=True, slots=True)
class DemoPaths:
    root: Path

    @property
    def dataset_root(self) -> Path:
        return self.root / "dataset"

    @property
    def db_path(self) -> Path:
        return self.root / "db"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def codebook_path(self) -> Path:
        return self.artifacts / "codebook.npz"

    @property
    def descriptor_cache(self) -> Path:
        return self.artifacts / "cache"

    @property
    def vector_storage(self) -> Path:
        return self.artifacts / "vectors"

    @property
    def manifest_path(self) -> Path:
        return self.artifacts / "training_manifest.jsonl"

    @property
    def staging_root(self) -> Path:
        return self.dataset_root / DEFAULT_STAGING_DIR

    @property
    def sample_query(self) -> Path:
        return self.root / "sample_query.wav"

    @property
    def marker(self) -> Path:
        return self.root / MARKER

    def index_config(self) -> CopyrightIndexConfig:
        return CopyrightIndexConfig(
            self.dataset_root, self.codebook_path, self.descriptor_cache, self.vector_storage
        )

    def env(self) -> dict[str, str]:
        """Variables ``COPYRIGHT_*`` obligatorias para arrancar el servidor sobre esta demo."""
        return {
            DB_PATH: str(self.db_path),
            DATASET_ROOT: str(self.dataset_root),
            CODEBOOK_PATH: str(self.codebook_path),
            DESCRIPTOR_CACHE: str(self.descriptor_cache),
            VECTOR_STORAGE: str(self.vector_storage),
        }


def synthesize(song: DemoSong) -> np.ndarray:
    """Audio determinista: cada bloque de 5 s es un acorde distinto alrededor de ``base_hz``."""
    rng = np.random.default_rng(song.seed)
    samples = round(song.seconds * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    signal = np.zeros(samples)
    block = round(5.0 * SAMPLE_RATE)
    for start in range(0, samples, block):
        piece = slice(start, start + block)
        for ratio in rng.uniform(1.0, 6.0, size=3):
            signal[piece] += 0.2 * np.sin(2 * np.pi * song.base_hz * ratio * t[piece])
    signal += 0.02 * rng.standard_normal(samples)
    return signal


def write_song(song: DemoSong, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, synthesize(song), SAMPLE_RATE)


def is_prepared(paths: DemoPaths) -> bool:
    """La demo está completa (marca escrita) y su HNSW se reabre sin reconstruirse."""
    try:
        marker = json.loads(paths.marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(marker, dict) or marker.get("version") != MARKER_VERSION:
        return False
    # Catalog(...) crearía una base vacía si db/ no existiera.
    if not paths.sample_query.is_file() or not paths.db_path.is_dir():
        return False
    try:
        catalog = open_catalog(paths.db_path, paths.index_config())
    except CopyrightIndexError:
        return False
    try:
        require_index(catalog)
        return len(CopyrightCatalog(catalog).list_songs()) == len(SONGS)
    except CopyrightIndexError:
        return False
    finally:
        catalog.close()


def prepare(root: Path) -> tuple[DemoPaths, bool]:
    """Prepara (o reutiliza) la demo en ``root``. Devuelve las rutas y si se reutilizó."""
    paths = DemoPaths(Path(root).absolute())
    if paths.root.exists():
        if not paths.root.is_dir():
            raise DemoError(f"{paths.root} existe y no es un directorio")
        if is_prepared(paths):
            return paths, True
        if any(paths.root.iterdir()):
            raise DemoError(
                f"{paths.root} ya existe y no contiene una demo completa. "
                "Bórralo a mano o usa otro --root; la herramienta no borra directorios."
            )

    for song in SONGS:
        write_song(song, paths.dataset_root / "tracks" / song.filename)
    paths.staging_root.mkdir(parents=True, exist_ok=True)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    catalog = Catalog(paths.db_path)
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        for song in SONGS:
            metadata = SongMetadata(
                song.song_id, song.title, song.artist, f"tracks/{song.filename}"
            )
            ingest_song(db, paths.dataset_root, metadata, song.first_segment_id)
        train_codebook(db, paths.index_config(), paths.manifest_path)
    finally:
        catalog.close()

    catalog = open_catalog(paths.db_path, paths.index_config())
    try:
        ensure_index(catalog, paths.index_config())
        require_index(catalog)
    finally:
        catalog.close()

    write_song(QUERY_SOURCE, paths.sample_query)
    # La marca va al final: sin ella, una preparación interrumpida no se reutiliza.
    paths.marker.write_text(
        json.dumps(
            {"version": MARKER_VERSION, "songs": [s.title for s in SONGS], "query": "Alpha"},
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths, False


def summary(paths: DemoPaths, reused: bool) -> str:
    env = paths.env()
    lines = [
        "Demo reutilizada (ya estaba preparada)" if reused else "Demo preparada",
        f"DB: {paths.db_path}",
        f"Dataset: {paths.dataset_root}",
        f"Codebook: {paths.codebook_path}",
        f"Descriptor cache: {paths.descriptor_cache}",
        f"Vector storage: {paths.vector_storage}",
        f"Query: {paths.sample_query}",
        "",
        "Bash:",
        *(f"export {name}={shlex.quote(value)}" for name, value in env.items()),
        "",
        "PowerShell:",
        *(f'$env:{name}="{value}"' for name, value in env.items()),
        "",
        "Servidor:",
        "uv run --group copyright-api python -m applications.copyright_detection.runtime",
        "",
        "Consulta:",
        f'curl -F "file=@{paths.sample_query}" http://127.0.0.1:8000/api/v1/analyze',
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m applications.copyright_detection.demo",
        description="Prepara una demo sintética de la API de copyright.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare", help="genera, ingesta, entrena e indexa")
    prepare_cmd.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"directorio de la demo (por defecto {DEFAULT_ROOT}, relativo al cwd)",
    )
    args = parser.parse_args(argv)
    try:
        paths, reused = prepare(args.root)
    except DemoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(
            f"error: la preparación falló y {Path(args.root).absolute()} puede quedar "
            "incompleto; bórralo a mano antes de repetir.",
            file=sys.stderr,
        )
        raise
    print(summary(paths, reused))
    return 0


if __name__ == "__main__":
    sys.exit(main())
