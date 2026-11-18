"""Catálogo de canciones originales para el API: lectura y audio de referencia.

Solo lectura. El catálogo se construye **antes** de arrancar el servidor
(``fma_catalog prepare`` o la demo): con el HNSW creado el motor bloquea
``INSERT``/``DELETE`` en ``audio_segments``, así que el runtime no añade ni quita
canciones.

Audio de referencia: el archivo se resuelve **solo** a partir de un ``song_id``
registrado. La ruta sale de ``songs.audio_path`` (relativa a la raíz del
dataset) y se valida con :func:`~applications.copyright_detection.ingestion.resolve_dataset_path`
(relativa, canónica, sin ``..`` ni enlaces que salgan de la raíz, archivo
existente). El cliente nunca aporta rutas: el endpoint recibe un entero.

Concurrencia: comparte el ``lock`` del analizador porque el ``Catalog`` no es
thread-safe. No abre ni cierra el ``Catalog``.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from applications.copyright_detection.catalog import AudioSegment, CopyrightCatalog, Song
from applications.copyright_detection.ingestion import AudioIngestionError, resolve_dataset_path
from engine.common.catalog import Catalog

logger = logging.getLogger(__name__)

#: Tipos MIME de los formatos que acepta el pipeline; el resto no se sirve.
AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".au": "audio/basic",
}
MAX_SONG_ID = 2**31 - 1


@dataclass(frozen=True, slots=True)
class SongEntry:
    song: Song
    segments: tuple[AudioSegment, ...]


@dataclass(frozen=True, slots=True)
class ReferenceAudio:
    path: Path
    media_type: str
    #: Nombre para ``Content-Disposition``: solo el último componente de ``audio_path``.
    filename: str


class CatalogLibrary:
    """Consultas de solo lectura sobre ``songs``/``audio_segments`` y su audio."""

    def __init__(
        self, catalog: Catalog, dataset_root: str | Path, lock: "threading.RLock | None" = None
    ) -> None:
        self._db = CopyrightCatalog(catalog)
        self._dataset_root = Path(dataset_root)
        self._lock = lock if lock is not None else threading.RLock()

    def list_songs(self) -> list[SongEntry]:
        with self._lock:
            return [
                SongEntry(song, tuple(self._db.list_segments(song.song_id)))
                for song in self._db.list_songs()
            ]

    def get_song(self, song_id: int) -> SongEntry | None:
        if not _valid_song_id(song_id):
            return None
        with self._lock:
            song = self._db.get_song(song_id)
            if song is None:
                return None
            return SongEntry(song, tuple(self._db.list_segments(song_id)))

    def reference_audio(self, song_id: int) -> ReferenceAudio | None:
        """Archivo original de ``song_id`` validado bajo la raíz, o ``None`` si no se sirve."""
        entry = self.get_song(song_id)
        if entry is None:
            return None
        audio_path = entry.song.audio_path
        media_type = AUDIO_MEDIA_TYPES.get(Path(audio_path).suffix.lower())
        if media_type is None:
            logger.warning("song_id=%d tiene un formato no servible: %r", song_id, audio_path)
            return None
        try:
            path = resolve_dataset_path(self._dataset_root, audio_path)
        except AudioIngestionError as exc:
            # Ruta que sale de la raíz, no canónica o archivo ausente: no se sirve.
            logger.warning("no se sirve el audio de song_id=%d: %s", song_id, exc)
            return None
        return ReferenceAudio(path, media_type, Path(audio_path).name)


def _valid_song_id(song_id: object) -> bool:
    return type(song_id) is int and 0 < song_id <= MAX_SONG_ID
