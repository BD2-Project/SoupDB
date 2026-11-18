"""Ingesta de una canción original del dataset de copyright en el catálogo relacional.

Pasos: valida la ruta bajo la raíz del dataset, decodifica el audio con el
pipeline de la Parte 4 (``soupdb_vector.multimedia.audio_features``), lo segmenta
con ``iter_audio_segments`` y registra la canción y sus segmentos a través de
:class:`~applications.copyright_detection.catalog.CopyrightCatalog`.

No calcula MFCC, vectores ni índices HNSW: solo llena ``songs`` y ``audio_segments``.

Limitaciones conocidas:

- La API pública del plugin no devuelve a la vez la señal completa y los
  segmentos, así que el archivo se decodifica dos veces: una con ``load_audio``
  (duración real) y otra con ``iter_audio_segments`` (segmentos).
- Todas las comprobaciones se hacen antes del primer ``INSERT``, pero la
  escritura de canción + segmentos **no es atómica**: si una inserción falla a
  mitad (o otro proceso escribe a la vez), puede quedar la canción con parte de
  sus segmentos. Se resolverá al integrar la ingesta con transacciones.

Requiere ``soupdb-vector`` (librosa); por eso ``app.py`` no importa este módulo.
"""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import soundfile
from soupdb_vector.multimedia.audio_features import (
    SAMPLE_RATE,
    iter_audio_segments,
    load_audio,
    segment_id,
)
from soupdb_vector.multimedia.audio_features import (
    SEGMENT_SECONDS as PIPELINE_SEGMENT_SECONDS,
)

from applications.copyright_detection.catalog import (
    AudioSegment,
    CopyrightCatalog,
    Song,
    _require_positive_id,
    _require_text,
)
from applications.copyright_detection.config import SEGMENT_SECONDS

if SEGMENT_SECONDS != PIPELINE_SEGMENT_SECONDS:
    raise RuntimeError(
        f"config.SEGMENT_SECONDS={SEGMENT_SECONDS} no coincide con el pipeline de audio "
        f"de soupdb-vector ({PIPELINE_SEGMENT_SECONDS})"
    )

#: Mayor valor de una columna INT de SoupDB (entero de 32 bits con signo). Se
#: comprueba antes de escribir para no fallar a mitad de la ingesta.
_MAX_INT = 2**31 - 1

#: Errores de decodificación de librosa: soundfile siempre; audioread solo si
#: está instalado (el lock lo instala para Python < 3.12). Público: la búsqueda
#: de candidatos decodifica igual y traduce los mismos errores.
AUDIO_DECODE_ERRORS: tuple[type[Exception], ...] = (soundfile.SoundFileError, EOFError)
try:
    from audioread.exceptions import DecodeError as _AudioreadDecodeError
except ImportError:
    pass
else:
    AUDIO_DECODE_ERRORS += (_AudioreadDecodeError,)


class AudioIngestionError(Exception):
    """La canción no se pudo ingestar; el mensaje indica la causa y el archivo."""


@dataclass(frozen=True, slots=True)
class SongMetadata:
    """Datos de entrada de una canción. ``audio_path`` es relativo a la raíz del dataset."""

    song_id: int
    title: str
    artist: str
    audio_path: str

    def __post_init__(self) -> None:
        _require_positive_id("song_id", self.song_id)
        _require_text("title", self.title)
        _require_text("artist", self.artist)
        _require_text("audio_path", self.audio_path)


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Canción y segmentos registrados."""

    song: Song
    segments: tuple[AudioSegment, ...]


def resolve_dataset_path(dataset_root: str | Path, audio_path: str) -> Path:
    """Ruta absoluta de ``audio_path`` dentro de ``dataset_root``, validada.

    Rechaza rutas absolutas, con ``\\`` o ``#`` (separador de segmento), que
    salgan de la raíz al resolverse (``..`` o enlaces), que no existan, que no
    sean archivos o que no estén en forma canónica (``a/./b``, ``a//b``,
    ``a/../b``): ``audio_path`` forma parte de ``audio_ref`` y debe ser único.
    """
    if (
        PurePosixPath(audio_path).is_absolute()
        or PureWindowsPath(audio_path).is_absolute()
        or PureWindowsPath(audio_path).drive
    ):
        raise AudioIngestionError(f"audio_path debe ser relativo a la raíz: {audio_path!r}")
    if "\\" in audio_path:
        raise AudioIngestionError(f"audio_path debe usar '/' como separador: {audio_path!r}")
    if "#" in audio_path:
        raise AudioIngestionError(
            f"audio_path no puede contener '#' (separa el segmento en audio_ref): {audio_path!r}"
        )

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise AudioIngestionError(f"la raíz del dataset no es un directorio: {root}")
    candidate = (root / audio_path).resolve()
    if not candidate.is_relative_to(root):
        raise AudioIngestionError(f"audio_path sale de la raíz del dataset: {audio_path!r}")
    if not candidate.exists():
        raise AudioIngestionError(f"el archivo no existe: {audio_path!r} (raíz {root})")
    if not candidate.is_file():
        raise AudioIngestionError(f"audio_path no es un archivo: {audio_path!r}")

    parts = PurePosixPath(audio_path).parts
    if PurePosixPath(audio_path).as_posix() != audio_path or any(p in (".", "..") for p in parts):
        canonical = candidate.relative_to(root).as_posix()
        raise AudioIngestionError(
            f"audio_path no está en forma canónica: {audio_path!r} (usar {canonical!r})"
        )
    return candidate


def ingest_song(
    catalog: CopyrightCatalog,
    dataset_root: str | Path,
    metadata: SongMetadata,
    first_segment_id: int,
) -> IngestionResult:
    """Ingesta una canción: valida, segmenta y registra canción y segmentos.

    Los segmentos reciben ``segment_id = first_segment_id + segment_index``. Todas
    las validaciones (ruta, audio, ids, duplicados) ocurren antes de escribir; si
    alguna falla no se inserta nada. La escritura posterior no es atómica (ver
    docstring del módulo).
    """
    _require_positive_id("first_segment_id", first_segment_id)
    path = resolve_dataset_path(dataset_root, metadata.audio_path)

    if catalog.has_vector_index():
        raise AudioIngestionError(
            "audio_segments ya tiene índice vectorial: el motor bloquea nuevas filas fuera "
            "de una transacción; ingestar todo el catálogo antes de crear el índice"
        )
    if catalog.get_song(metadata.song_id) is not None:
        raise AudioIngestionError(f"la canción song_id={metadata.song_id} ya está registrada")
    same_path = [s.song_id for s in catalog.list_songs() if s.audio_path == metadata.audio_path]
    if same_path:
        raise AudioIngestionError(
            f"{metadata.audio_path!r} ya está registrado como song_id={same_path[0]}"
        )

    try:
        duration_seconds = len(load_audio(path)) / SAMPLE_RATE
        pieces = [(index, len(samples)) for index, samples in iter_audio_segments(path)]
    except AUDIO_DECODE_ERRORS as exc:
        raise AudioIngestionError(f"no se pudo decodificar {metadata.audio_path!r}: {exc}") from exc
    if not pieces:
        raise AudioIngestionError(
            f"{metadata.audio_path!r} no produce ningún segmento indexable "
            f"(dura {duration_seconds:.3f} s); no se registra"
        )

    song = Song(
        metadata.song_id, metadata.title, metadata.artist, metadata.audio_path, duration_seconds
    )
    segments = tuple(
        AudioSegment(
            segment_id=first_segment_id + index,
            song_id=song.song_id,
            segment_index=index,
            start_seconds=index * SEGMENT_SECONDS,
            end_seconds=index * SEGMENT_SECONDS + length / SAMPLE_RATE,
            audio_ref=segment_id(song.audio_path, index),
        )
        for index, length in pieces
    )

    last_id = segments[-1].segment_id
    if last_id > _MAX_INT:
        raise AudioIngestionError(
            f"segment_id {last_id} supera el máximo de INT de SoupDB ({_MAX_INT})"
        )
    taken = [s.segment_id for s in segments if catalog.get_segment(s.segment_id) is not None]
    if taken:
        raise AudioIngestionError(
            f"los segment_id {taken} ya existen; elegir otro first_segment_id "
            f"(se necesitan {first_segment_id}..{last_id})"
        )
    if catalog.list_segments(song.song_id):
        raise AudioIngestionError(f"ya hay segmentos registrados para song_id={song.song_id}")

    catalog.add_song(song)
    for segment in segments:
        catalog.add_segment(segment)
    return IngestionResult(song=song, segments=segments)
