"""Servido de documentos multimedia del gestor."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from engine.common.errors import RecordNotFound, UnsupportedOperation

#: Tipos MIME que el gestor reconoce por extensión; el resto es binario opaco.
MEDIA_TYPES = {
    # imagen
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    # audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".au": "audio/basic",
    ".m4a": "audio/mp4",
    # vídeo
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    # texto / documentos
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}

DEFAULT_MEDIA_TYPE = "application/octet-stream"

#: Tamaño de cada trozo enviado por el socket: el archivo no se carga entero.
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB


def media_type_for(path: str | Path) -> str:
    """MIME por extensión (sin distinguir mayúsculas); desconocida → binario."""
    return MEDIA_TYPES.get(Path(path).suffix.lower(), DEFAULT_MEDIA_TYPE)


@dataclass(frozen=True, slots=True)
class DocumentHandle:
    """Documento resuelto; se lee en trozos para no cargarlo completo."""

    media_type: str
    size: int
    path: Path
    #: Solo el último componente, para ``Content-Disposition`` en el cliente.
    filename: str

    def read_chunks(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
        with self.path.open("rb") as file:
            while chunk := file.read(chunk_size):
                yield chunk


class DocumentStore:
    """Resuelve referencias relativas contra una raíz única del despliegue.

    La raíz es configuración del proceso (``SOUPDB_DOCUMENT_ROOT``), no un valor
    guardado por fila: en local apunta a la carpeta del usuario y en un contenedor
    al volumen montado. El código de resolución es idéntico en ambos casos.
    """

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise UnsupportedOperation(f"document root is not a directory: {resolved}")
        self._root = resolved

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, reference: str) -> DocumentHandle:
        """Devuelve el archivo real de ``reference`` o lanza ``RecordNotFound``."""
        if not isinstance(reference, str) or not reference.strip():
            raise RecordNotFound("document reference must be a non-empty string")
        if (
            PurePosixPath(reference).is_absolute()
            or PureWindowsPath(reference).is_absolute()
            or PureWindowsPath(reference).drive
        ):
            raise RecordNotFound(f"document reference must be relative: {reference!r}")
        if "\\" in reference:
            raise RecordNotFound(f"document reference must use '/' separators: {reference!r}")
        parts = PurePosixPath(reference).parts
        if not parts or any(part in (".", "..") for part in parts):
            raise RecordNotFound(f"document reference is not canonical: {reference!r}")
        if PurePosixPath(reference).as_posix() != reference:
            raise RecordNotFound(f"document reference is not canonical: {reference!r}")

        candidate = (self._root / reference).resolve()
        if not candidate.is_relative_to(self._root):
            raise RecordNotFound(f"document reference escapes the root: {reference!r}")
        if not candidate.exists():
            raise RecordNotFound(f"document does not exist: {reference!r}")
        if not candidate.is_file():
            raise RecordNotFound(f"document is not a file: {reference!r}")
        return DocumentHandle(
            media_type=media_type_for(candidate),
            size=candidate.stat().st_size,
            path=candidate,
            filename=candidate.name,
        )
