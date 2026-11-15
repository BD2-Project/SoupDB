"""Catálogo real de canciones a partir de Free Music Archive (FMA Small).

Uso::

    uv run --group copyright-api python -m applications.copyright_detection.fma_catalog prepare \\
        --audio-root D:/datasets/fma_small \\
        --tracks-csv D:/datasets/fma_metadata/tracks.csv \\
        --output-root .demo/fma --per-genre 10

    uv run --group copyright-api python -m applications.copyright_detection.fma_catalog \\
        make-query --audio-root D:/datasets/fma_small --output-root .demo/fma

Capa adicional sobre las APIs existentes de la aplicación; no reimplementa
MFCC, K-Means, TF-IDF ni HNSW. La demo sintética (:mod:`.demo`) sigue siendo la
base de los tests deterministas. El dataset no se descarga ni se copia: se lee
de ``--audio-root`` (la carpeta que contiene ``000/``, ``001/``, ...) y esa
misma carpeta es el ``dataset_root`` del perfil.

Flujo::

    tracks.csv (subset small) → selección balanceada por genre_top
    → CopyrightCatalog.initialize() → ingest_song() por pista
    → train_codebook(K=128) → open_catalog() → ensure_index(HNSW euclidean)
    → require_index() → verificación de conteos → catalog_manifest.json

Selección: por cada género, las pistas se ordenan por ``track_id`` y se toman
las primeras ``per_genre`` que existen en disco **y** se ingestan. Una pista
cuyo MP3 falta o no se decodifica se registra en ``skipped_tracks`` y se prueba
la siguiente del mismo género. ``ingest_song`` valida y decodifica antes del
primer ``INSERT``, así que una pista descartada no deja filas; si aun así
quedaran filas de una pista fallida, la preparación aborta.

Estructura creada en ``--output-root`` (los MP3 originales no se copian)::

    db/                                 base SoupDB (songs, audio_segments)
    artifacts/codebook.npz              codebook K=128 + IDF
    artifacts/cache/                    caché de MFCC
    artifacts/vectors/                  HNSW persistente
    artifacts/training_manifest.jsonl   manifest de entrenamiento
    catalog_manifest.json               resumen; se escribe al final (marca de completo)
    sample_query.mp3                    solo con ``make-query``; no se registra en SoupDB
"""

import argparse
import csv
import json
import re
import shlex
import shutil
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from applications.copyright_detection.catalog import CopyrightCatalog
from applications.copyright_detection.config import METRIC, SEGMENT_SECONDS
from applications.copyright_detection.indexing import (
    CODEBOOK_K,
    INDEX_NAME,
    CopyrightIndexConfig,
    CopyrightIndexError,
    ensure_index,
    open_catalog,
    require_index,
    train_codebook,
)
from applications.copyright_detection.ingestion import (
    AudioIngestionError,
    IngestionResult,
    SongMetadata,
    ingest_song,
)
from applications.copyright_detection.settings import (
    CODEBOOK_PATH,
    DATASET_ROOT,
    DB_PATH,
    DESCRIPTOR_CACHE,
    VECTOR_STORAGE,
)
from engine.common.catalog import Catalog

DATASET_NAME = "FMA Small"
DATASET_SOURCE = "Free Music Archive"
SMALL_SUBSET = "small"
#: Los 8 géneros (``track/genre_top``) balanceados de FMA Small.
FMA_SMALL_GENRES = (
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
)
DEFAULT_PER_GENRE = 10
DEFAULT_OUTPUT_ROOT = Path(".demo/fma")
MANIFEST_NAME = "catalog_manifest.json"
MANIFEST_VERSION = 1
SAMPLE_QUERY_NAME = "sample_query.mp3"
#: Mayor ``song_id`` admitido (columna INT de SoupDB, 32 bits con signo).
MAX_TRACK_ID = 2**31 - 1

#: Columnas de ``tracks.csv`` (cabecera doble: nivel 0, nivel 1).
COL_SUBSET = ("set", "subset")
COL_TITLE = ("track", "title")
COL_ARTIST = ("artist", "name")
COL_GENRE = ("track", "genre_top")
COL_LICENSE = ("track", "license")
REQUIRED_COLUMNS = (COL_SUBSET, COL_TITLE, COL_ARTIST, COL_GENRE)

SKIP_MISSING = "missing_file"
SKIP_UNDECODABLE = "ingestion_failed"


class FmaCatalogError(Exception):
    """El catálogo FMA no se puede preparar; el mensaje indica la causa."""


@dataclass(frozen=True, slots=True)
class FmaTrack:
    """Pista de FMA Small. ``source_path`` es relativo a la raíz de audio (``000/000002.mp3``)."""

    track_id: int
    title: str
    artist: str
    genre: str
    source_path: str
    license: str = ""


@dataclass(frozen=True, slots=True)
class SkippedTrack:
    track_id: int
    genre: str
    audio_path: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class SelectedTrack:
    track_id: int
    song_id: int
    title: str
    artist: str
    genre: str
    audio_path: str
    duration_seconds: float
    segment_count: int
    first_segment_id: int
    last_segment_id: int
    license: str = ""


@dataclass(frozen=True, slots=True)
class FmaPaths:
    output_root: Path
    audio_root: Path

    @property
    def db_path(self) -> Path:
        return self.output_root / "db"

    @property
    def artifacts(self) -> Path:
        return self.output_root / "artifacts"

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
    def training_manifest(self) -> Path:
        return self.artifacts / "training_manifest.jsonl"

    @property
    def manifest(self) -> Path:
        return self.output_root / MANIFEST_NAME

    @property
    def sample_query(self) -> Path:
        return self.output_root / SAMPLE_QUERY_NAME

    def index_config(self) -> CopyrightIndexConfig:
        return CopyrightIndexConfig(
            self.audio_root, self.codebook_path, self.descriptor_cache, self.vector_storage
        )

    def env(self) -> dict[str, str]:
        """Variables ``COPYRIGHT_*`` obligatorias para arrancar el servidor sobre este catálogo."""
        return {
            DB_PATH: str(self.db_path),
            DATASET_ROOT: str(self.audio_root),
            CODEBOOK_PATH: str(self.codebook_path),
            DESCRIPTOR_CACHE: str(self.descriptor_cache),
            VECTOR_STORAGE: str(self.vector_storage),
        }


@dataclass(frozen=True, slots=True)
class FmaCatalogResult:
    paths: FmaPaths
    manifest: dict
    reused: bool


# --------------------------------------------------------------------------- metadata


def fma_audio_relative_path(track_id: int) -> str:
    """Ruta oficial de FMA relativa a la raíz de audio: ``2`` → ``000/000002.mp3``."""
    if isinstance(track_id, bool) or not isinstance(track_id, int):
        raise ValueError(f"track_id debe ser un entero, no {track_id!r}")
    if not 0 < track_id <= MAX_TRACK_ID:
        raise ValueError(f"track_id debe estar entre 1 y {MAX_TRACK_ID}: {track_id}")
    name = f"{track_id:06d}"
    return f"{name[:3]}/{name}.mp3"


def _require_per_genre(per_genre: object) -> int:
    if isinstance(per_genre, bool) or not isinstance(per_genre, int) or per_genre < 1:
        raise FmaCatalogError(f"per_genre debe ser un entero positivo, no {per_genre!r}")
    return per_genre


def _header_keys(level0: list[str], level1: list[str]) -> list[tuple[str, str]]:
    """Claves ``(nivel0, nivel1)``; un nivel 0 vacío hereda el anterior (como pandas)."""
    keys: list[tuple[str, str]] = []
    current = ""
    for index in range(max(len(level0), len(level1))):
        top = level0[index].strip() if index < len(level0) else ""
        sub = level1[index].strip() if index < len(level1) else ""
        if top:
            current = top
        keys.append((current, sub))
    return keys


def read_fma_small_tracks(tracks_csv: str | Path) -> list[FmaTrack]:
    """Lee ``tracks.csv`` de FMA (cabecera doble) y devuelve las pistas ``set/subset == small``.

    Formato oficial (``pandas.read_csv(..., index_col=0, header=[0, 1])``): fila 0
    con el grupo (``track``, ``artist``, ``set``...), fila 1 con el campo y, opcional,
    una fila ``track_id,,,...`` con el nombre del índice. La primera columna es el
    ``track_id``. Se lee con ``csv`` de la stdlib (campos con saltos de línea incluidos).
    Devuelve las pistas ordenadas por ``track_id``.
    """
    path = Path(tracks_csv)
    if not path.is_file():
        raise FmaCatalogError(f"no existe el archivo de metadata {path}")
    tracks: list[FmaTrack] = []
    seen: set[int] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            level0 = next(reader)
            level1 = next(reader)
        except StopIteration:
            raise FmaCatalogError(
                f"{path} no tiene la cabecera doble de FMA tracks.csv (está vacío o incompleto)"
            ) from None
        keys = _header_keys(level0, level1)
        columns = {key: index for index, key in enumerate(keys) if index > 0}
        missing = [f"{a}/{b}" for a, b in REQUIRED_COLUMNS if (a, b) not in columns]
        if missing:
            raise FmaCatalogError(
                f"{path} no tiene las columnas {missing}; ¿es tracks.csv de fma_metadata?"
            )
        license_col = columns.get(COL_LICENSE)

        def cell(row: list[str], key: tuple[str, str]) -> str:
            index = columns[key]
            return row[index].strip() if index < len(row) else ""

        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue
            first = row[0].strip()
            if first == "track_id" and not any(field.strip() for field in row[1:]):
                continue  # fila con el nombre del índice
            try:
                track_id = int(first)
            except ValueError:
                raise FmaCatalogError(
                    f"{path}, línea {reader.line_num}: track_id inválido {first!r}"
                ) from None
            if cell(row, COL_SUBSET) != SMALL_SUBSET:
                continue
            try:
                source_path = fma_audio_relative_path(track_id)
            except ValueError as exc:
                raise FmaCatalogError(f"{path}, línea {reader.line_num}: {exc}") from None
            if track_id in seen:
                raise FmaCatalogError(f"{path}: track_id {track_id} repetido")
            seen.add(track_id)
            license_ = ""
            if license_col is not None and license_col < len(row):
                license_ = row[license_col].strip()
            tracks.append(
                FmaTrack(
                    track_id=track_id,
                    title=cell(row, COL_TITLE) or f"FMA track {track_id}",
                    artist=cell(row, COL_ARTIST) or "Unknown artist",
                    genre=cell(row, COL_GENRE),
                    source_path=source_path,
                    license=license_,
                )
            )
    if not tracks:
        raise FmaCatalogError(f"{path} no contiene pistas del subset {SMALL_SUBSET!r}")
    tracks.sort(key=lambda track: track.track_id)
    return tracks


def validate_audio_root(audio_root: str | Path) -> Path:
    """Raíz de audio FMA (con carpetas ``000/``, ``001/``...), resuelta a ruta absoluta."""
    root = Path(audio_root).resolve()
    if not root.is_dir():
        raise FmaCatalogError(f"la raíz de audio FMA no es un directorio: {root}")
    has_shards = any(
        child.is_dir() and len(child.name) == 3 and child.name.isdigit() for child in root.iterdir()
    )
    if not has_shards:
        hint = ""
        if (root / "fma_small").is_dir():
            hint = f"; ¿quisiste decir {root / 'fma_small'}?"
        raise FmaCatalogError(
            f"{root} no tiene la estructura de FMA (carpetas 000/, 001/, ...){hint}"
        )
    return root


def group_by_genre(
    tracks: Sequence[FmaTrack], genres: Sequence[str] = FMA_SMALL_GENRES
) -> dict[str, list[FmaTrack]]:
    """Pistas de cada género de ``genres``, ordenadas por ``track_id``. Falla si falta alguno."""
    if not genres:
        raise FmaCatalogError("la lista de géneros está vacía")
    grouped: dict[str, list[FmaTrack]] = {genre: [] for genre in genres}
    for track in tracks:
        if track.genre in grouped:
            grouped[track.genre].append(track)
    absent = [genre for genre, items in grouped.items() if not items]
    if absent:
        raise FmaCatalogError(f"la metadata no tiene pistas del subset small para {absent}")
    for items in grouped.values():
        items.sort(key=lambda track: track.track_id)
    return grouped


def _existing(
    candidates: Sequence[FmaTrack], audio_root: Path, skipped: list[SkippedTrack]
) -> Iterator[FmaTrack]:
    """Candidatas con MP3 en disco, en orden; las ausentes se anotan en ``skipped``."""
    for track in candidates:
        if (audio_root / track.source_path).is_file():
            yield track
        else:
            skipped.append(
                SkippedTrack(
                    track.track_id,
                    track.genre,
                    track.source_path,
                    SKIP_MISSING,
                    "el MP3 no existe bajo la raíz de audio",
                )
            )


def _not_enough(genre: str, found: int, per_genre: int, available: int) -> FmaCatalogError:
    return FmaCatalogError(
        f"el género {genre!r} solo tiene {found} pistas válidas de {per_genre} pedidas "
        f"({available} en la metadata); baja --per-genre o completa el dataset"
    )


def select_fma_small_tracks(
    tracks_csv: str | Path,
    audio_root: str | Path,
    per_genre: int = DEFAULT_PER_GENRE,
    genres: Sequence[str] = FMA_SMALL_GENRES,
) -> list[FmaTrack]:
    """Selección balanceada y determinista: las primeras ``per_genre`` pistas por ``track_id``
    de cada género cuyo MP3 existe. Ordenada por género (en el orden de ``genres``) y
    ``track_id``. No decodifica audio: :func:`prepare_fma_catalog` además descarta las
    pistas que no se pueden ingestar.
    """
    per_genre = _require_per_genre(per_genre)
    root = validate_audio_root(audio_root)
    grouped = group_by_genre(read_fma_small_tracks(tracks_csv), genres)
    selected: list[FmaTrack] = []
    for genre, candidates in grouped.items():
        chosen: list[FmaTrack] = []
        for track in _existing(candidates, root, []):
            chosen.append(track)
            if len(chosen) == per_genre:
                break
        if len(chosen) < per_genre:
            raise _not_enough(genre, len(chosen), per_genre, len(candidates))
        selected.extend(chosen)
    return selected


# --------------------------------------------------------------------------- preparación


def _rows_for(db: CopyrightCatalog, song_id: int) -> bool:
    return db.get_song(song_id) is not None or bool(db.list_segments(song_id))


def _try_ingest(
    db: CopyrightCatalog, audio_root: Path, track: FmaTrack, first_segment_id: int
) -> IngestionResult | str:
    """Ingesta ``track``; si falla sin haber escrito nada, devuelve el motivo."""
    metadata = SongMetadata(track.track_id, track.title, track.artist, track.source_path)
    try:
        return ingest_song(db, audio_root, metadata, first_segment_id)
    except Exception as exc:
        if _rows_for(db, track.track_id):
            raise FmaCatalogError(
                f"la ingesta de track_id={track.track_id} falló tras escribir filas: {exc}"
            ) from exc
        return _safe_error_detail(exc, track.source_path)


#: Rutas absolutas dentro de mensajes de excepción: entre comillas (pueden tener
#: espacios) o sueltas. Unidad Windows (``C:\``, ``C:/``), UNC (``\\server``) o POSIX
#: (``/tmp/...``). Las barras pueden venir duplicadas por ``repr`` (``C:\\Users``).
_ABSOLUTE = r"(?:[A-Za-z]:[\\/]|\\\\|/(?=[^\s'\"/]))"
_QUOTED_PATH = re.compile(rf"(['\"])({_ABSOLUTE}[^'\"]*)\1")
_BARE_PATH = re.compile(rf"(?<![\w.:/\\]){_ABSOLUTE}[^\s'\"]*")
REDACTED_PATH = "<path>"


def _safe_error_detail(exc: BaseException, relative_path: str) -> str:
    """Mensaje de ``exc`` apto para el manifest: sin rutas absolutas del sistema.

    Las librerías (libsndfile, audioread...) incluyen la ruta absoluta del archivo en
    sus mensajes, con formato distinto según la plataforma. Cada ruta absoluta se
    sustituye por ``relative_path`` si apunta a la pista (termina en ella) y por
    ``<path>`` en cualquier otro caso. Conserva el tipo de excepción cuando no es
    una :class:`AudioIngestionError` (cuyo mensaje ya describe la causa).
    """
    message = str(exc)
    if not isinstance(exc, AudioIngestionError):
        message = f"{type(exc).__name__}: {message}"

    def replacement(path: str) -> str:
        normalized = re.sub(r"[\\/]+", "/", path)
        if normalized == relative_path or normalized.endswith("/" + relative_path):
            return relative_path
        return REDACTED_PATH

    def bare(match: re.Match[str]) -> str:
        # Una ruta sin comillas no se lleva la puntuación que la sigue: "(/a/b)." → "(<path>).".
        path = match[0]
        core = path.rstrip(".,;:)]}>")
        return replacement(core) + path[len(core) :]

    message = _QUOTED_PATH.sub(lambda m: m[1] + replacement(m[2]) + m[1], message)
    return _BARE_PATH.sub(bare, message)


def _ingest_balanced(
    db: CopyrightCatalog,
    audio_root: Path,
    grouped: dict[str, list[FmaTrack]],
    per_genre: int,
) -> tuple[list[SelectedTrack], list[SkippedTrack]]:
    selected: list[SelectedTrack] = []
    skipped: list[SkippedTrack] = []
    next_segment_id = 1
    for genre, candidates in grouped.items():
        count = 0
        for track in _existing(candidates, audio_root, skipped):
            outcome = _try_ingest(db, audio_root, track, next_segment_id)
            if isinstance(outcome, str):
                skipped.append(
                    SkippedTrack(
                        track.track_id, genre, track.source_path, SKIP_UNDECODABLE, outcome
                    )
                )
                continue
            first, last = outcome.segments[0].segment_id, outcome.segments[-1].segment_id
            selected.append(
                SelectedTrack(
                    track_id=track.track_id,
                    song_id=outcome.song.song_id,
                    title=track.title,
                    artist=track.artist,
                    genre=genre,
                    audio_path=outcome.song.audio_path,
                    duration_seconds=round(outcome.song.duration_seconds, 3),
                    segment_count=len(outcome.segments),
                    first_segment_id=first,
                    last_segment_id=last,
                    license=track.license,
                )
            )
            next_segment_id = last + 1
            count += 1
            if count == per_genre:
                break
        if count < per_genre:
            raise _not_enough(genre, count, per_genre, len(candidates))
    return selected, skipped


def _verify_counts(catalog: Catalog, selected: Sequence[SelectedTrack]) -> int:
    db = CopyrightCatalog(catalog)
    songs = db.list_songs()
    if sorted(s.song_id for s in songs) != sorted(t.song_id for t in selected):
        raise FmaCatalogError(f"la base tiene {len(songs)} canciones; se esperaban {len(selected)}")
    segment_ids = [seg.segment_id for s in songs for seg in db.list_segments(s.song_id)]
    expected = sum(t.segment_count for t in selected)
    if len(segment_ids) != expected or len(set(segment_ids)) != expected:
        raise FmaCatalogError(
            f"la base tiene {len(segment_ids)} segmentos ({len(set(segment_ids))} ids "
            f"distintos); se esperaban {expected}"
        )
    return expected


def build_manifest(
    per_genre: int,
    genres: Sequence[str],
    selected: Sequence[SelectedTrack],
    skipped: Sequence[SkippedTrack],
) -> dict:
    """Contenido de ``catalog_manifest.json``; sin rutas absolutas."""
    return {
        "version": MANIFEST_VERSION,
        "dataset": DATASET_NAME,
        "source": DATASET_SOURCE,
        "number_of_songs": len(selected),
        "number_of_segments": sum(t.segment_count for t in selected),
        "genres": list(genres),
        "per_genre": per_genre,
        "segment_seconds": SEGMENT_SECONDS,
        "index": {"name": INDEX_NAME, "kind": "HNSW"},
        "metric": METRIC,
        "codebook_k": CODEBOOK_K,
        "selected_tracks": [asdict(track) for track in selected],
        "skipped_tracks": [asdict(track) for track in skipped],
    }


def read_manifest(output_root: str | Path) -> dict:
    path = Path(output_root) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise FmaCatalogError(f"no existe {path}; ejecuta antes 'prepare'") from None
    except ValueError as exc:
        raise FmaCatalogError(f"{path} no es JSON válido: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise FmaCatalogError(f"{path} no es un manifest de catálogo FMA reconocido")
    return manifest


def _reuse(paths: FmaPaths, per_genre: int, genres: Sequence[str]) -> dict:
    """Valida un catálogo ya preparado en ``paths.output_root`` y devuelve su manifest."""
    manifest = read_manifest(paths.output_root)
    if manifest.get("per_genre") != per_genre or manifest.get("genres") != list(genres):
        raise FmaCatalogError(
            f"{paths.output_root} ya tiene un catálogo con per_genre={manifest.get('per_genre')}; "
            "bórralo a mano o usa otro --output-root"
        )
    try:
        catalog = open_catalog(paths.db_path, paths.index_config())
    except CopyrightIndexError as exc:
        raise FmaCatalogError(f"no se pudo reabrir el catálogo existente: {exc}") from exc
    try:
        require_index(catalog)
        selected = [SelectedTrack(**track) for track in manifest["selected_tracks"]]
        _verify_counts(catalog, selected)
    except (CopyrightIndexError, KeyError, TypeError) as exc:
        raise FmaCatalogError(f"el catálogo existente no es válido: {exc}") from exc
    finally:
        catalog.close()
    return manifest


def prepare_fma_catalog(
    audio_root: str | Path,
    tracks_csv: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    per_genre: int = DEFAULT_PER_GENRE,
    *,
    seed: int = 0,
    genres: Sequence[str] = FMA_SMALL_GENRES,
) -> FmaCatalogResult:
    """Prepara (o reutiliza) un catálogo FMA Small balanceado en ``output_root``.

    ``seed`` solo afecta al entrenamiento K-Means; la selección es determinista.
    Nunca borra directorios: si ``output_root`` tiene otro contenido, falla.
    """
    per_genre = _require_per_genre(per_genre)
    root = validate_audio_root(audio_root)
    paths = FmaPaths(Path(output_root).absolute(), root)
    if paths.output_root.exists():
        if not paths.output_root.is_dir():
            raise FmaCatalogError(f"{paths.output_root} existe y no es un directorio")
        if paths.manifest.is_file():
            return FmaCatalogResult(paths, _reuse(paths, per_genre, genres), reused=True)
        if any(paths.output_root.iterdir()):
            raise FmaCatalogError(
                f"{paths.output_root} ya existe y no contiene un catálogo FMA completo. "
                "Bórralo a mano o usa otro --output-root; la herramienta no borra directorios."
            )
    if root == paths.output_root or root.is_relative_to(paths.output_root.resolve()):
        raise FmaCatalogError("--output-root no puede contener la raíz de audio FMA")

    grouped = group_by_genre(read_fma_small_tracks(tracks_csv), genres)
    for genre, candidates in grouped.items():
        if len(candidates) < per_genre:
            raise FmaCatalogError(
                f"el género {genre!r} solo tiene {len(candidates)} pistas en la metadata; "
                f"se piden {per_genre} (--per-genre)"
            )

    paths.artifacts.mkdir(parents=True, exist_ok=True)
    config = paths.index_config()
    catalog = Catalog(paths.db_path)
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        selected, skipped = _ingest_balanced(db, root, grouped, per_genre)
        train_codebook(db, config, paths.training_manifest, seed=seed)
    finally:
        catalog.close()

    catalog = open_catalog(paths.db_path, config)
    try:
        ensure_index(catalog, config)
        require_index(catalog)
        _verify_counts(catalog, selected)
    finally:
        catalog.close()

    manifest = build_manifest(per_genre, genres, selected, skipped)
    # El manifest va al final: sin él, una preparación interrumpida no se reutiliza.
    paths.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    return FmaCatalogResult(paths, manifest, reused=False)


def make_query(
    audio_root: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    track_id: int | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Copia el MP3 de una pista catalogada a ``output_root/sample_query.mp3``.

    Sin ``track_id`` usa la primera pista seleccionada. El archivo queda fuera de la
    raíz de audio y no se registra en SoupDB: es una consulta idéntica a una pista
    del catálogo (smoke test), no una evaluación de copyright. No sobrescribe un
    ``sample_query.mp3`` existente (puede ser un remix del usuario) salvo con
    ``overwrite``.
    """
    root = validate_audio_root(audio_root)
    paths = FmaPaths(Path(output_root).absolute(), root)
    tracks = read_manifest(paths.output_root).get("selected_tracks") or []
    if not tracks:
        raise FmaCatalogError(f"{paths.manifest} no tiene pistas seleccionadas")
    if track_id is None:
        track = tracks[0]
    else:
        matches = [t for t in tracks if t.get("track_id") == track_id]
        if not matches:
            raise FmaCatalogError(f"track_id {track_id} no está en el catálogo preparado")
        track = matches[0]
    source = root / track["audio_path"]
    if not source.is_file():
        raise FmaCatalogError(f"no existe {source}")
    if paths.sample_query.exists() and not overwrite:
        raise FmaCatalogError(f"{paths.sample_query} ya existe; usa --overwrite para reemplazarlo")
    shutil.copyfile(source, paths.sample_query)
    return paths.sample_query


# --------------------------------------------------------------------------- CLI


def summary(result: FmaCatalogResult) -> str:
    paths, manifest = result.paths, result.manifest
    env = paths.env()
    lines = [
        "FMA catalog reused (already prepared)" if result.reused else "FMA catalog prepared",
        f"Songs: {manifest['number_of_songs']}",
        f"Segments: {manifest['number_of_segments']}",
        f"Genres: {len(manifest['genres'])}",
        f"Skipped tracks: {len(manifest['skipped_tracks'])}",
        "HNSW: ready",
        f"DB: {paths.db_path}",
        f"Dataset root: {paths.audio_root}",
        f"Codebook: {paths.codebook_path}",
        f"Vectors: {paths.vector_storage}",
        f"Manifest: {paths.manifest}",
        "",
        "Bash:",
        *(f"export {name}={shlex.quote(value)}" for name, value in env.items()),
        "",
        "PowerShell:",
        *(f'$env:{name}="{value}"' for name, value in env.items()),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m applications.copyright_detection.fma_catalog",
        description="Prepara un catálogo real de canciones con FMA Small.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_cmd = commands.add_parser("prepare", help="selecciona, ingesta, entrena e indexa")
    prepare_cmd.add_argument("--audio-root", type=Path, required=True, help="carpeta fma_small/")
    prepare_cmd.add_argument(
        "--tracks-csv", type=Path, required=True, help="fma_metadata/tracks.csv"
    )
    prepare_cmd.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_cmd.add_argument("--per-genre", type=int, default=DEFAULT_PER_GENRE)
    prepare_cmd.add_argument("--seed", type=int, default=0, help="semilla de K-Means")

    query_cmd = commands.add_parser("make-query", help="copia una pista a sample_query.mp3")
    query_cmd.add_argument("--audio-root", type=Path, required=True, help="carpeta fma_small/")
    query_cmd.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    query_cmd.add_argument("--track-id", type=int, default=None)
    query_cmd.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "make-query":
        try:
            target = make_query(
                args.audio_root, args.output_root, args.track_id, overwrite=args.overwrite
            )
        except FmaCatalogError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Query: {target}")
        print(f'curl -F "file=@{target}" http://127.0.0.1:8000/api/v1/analyze')
        return 0

    try:
        result = prepare_fma_catalog(
            args.audio_root, args.tracks_csv, args.output_root, args.per_genre, seed=args.seed
        )
    except (FmaCatalogError, CopyrightIndexError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        output = Path(args.output_root).absolute()
        if output.is_dir() and any(output.iterdir()) and not (output / MANIFEST_NAME).exists():
            print(f"{output} quedó incompleto; bórralo a mano antes de repetir.", file=sys.stderr)
        return 1
    except Exception:
        print(
            f"error: la preparación falló y {Path(args.output_root).absolute()} puede quedar "
            "incompleto; bórralo a mano antes de repetir.",
            file=sys.stderr,
        )
        raise
    print(summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
