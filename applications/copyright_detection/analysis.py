"""Servicio de análisis: upload → staging único → búsqueda HNSW → evidencia → AnalysisResult.

FastAPI solo recibe el multipart, llama a :meth:`CopyrightAnalyzer.analyze` y
traduce errores. Este servicio:

1. Escribe el upload por bloques en ``<dataset_root>/<staging_dir>/<analysis_id>/``
   con un nombre generado en servidor (nunca el del cliente). ``analysis_id`` es
   un ``uuid4``: cada análisis usa una ruta nueva. Es obligatorio porque la caché
   de MFCC del perfil usa la ruta absoluta como clave, no el contenido; reutilizar
   una ruta devolvería descriptores de un audio anterior.
2. Llama a ``search_audio`` y ``build_temporal_evidence`` (sin copiar su lógica).
3. Convierte cada ``TemporalRun`` en un ``CopyrightMatch``.
4. Borra siempre el archivo y su carpeta del análisis (``try``/``finally``).

El candidato es solo una consulta: no se registra en ``songs``,
``audio_segments`` ni en el HNSW. La caché de MFCC conserva entradas de rutas
temporales ya borradas; no hay riesgo de reutilizarlas (la ruta no se repite) y
limpiarlas queda como mejora futura.

Concurrencia: el ``Catalog`` compartido no está documentado como thread-safe, así
que la búsqueda y la evidencia se serializan con un ``RLock`` del servicio. La
escritura del staging va fuera del lock. El servicio no abre ni cierra el
``Catalog``: lo hace quien lo creó con ``open_catalog``.

El resultado es evidencia de similitud acústica y temporal; no determina copyright.
"""

import logging
import re
import shutil
import threading
import uuid
from pathlib import PurePosixPath, PureWindowsPath
from typing import BinaryIO

from applications.copyright_detection.config import SEGMENT_SECONDS
from applications.copyright_detection.errors import AnalysisUnavailableError, InvalidAudioError
from applications.copyright_detection.evidence import (
    TemporalEvidenceResult,
    build_temporal_evidence,
)
from applications.copyright_detection.indexing import CopyrightIndexConfig
from applications.copyright_detection.schemas import AnalysisResult, CopyrightMatch
from applications.copyright_detection.search import (
    CandidateAudioError,
    SearchUnavailableError,
    search_audio,
)
from engine.common.catalog import Catalog

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_K = 5
DEFAULT_STAGING_DIR = "queries"
#: Extensiones que se conservan para el decodificador; el resto se escribe sin extensión.
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".au"})
_CHUNK_BYTES = 1024 * 1024
_FALLBACK_FILENAME = "upload"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def display_filename(raw: str | None) -> str:
    """Nombre del cliente apto para mostrar: solo el último componente, sin ruta.

    ``../../evil.wav`` y ``C:\\x\\evil.wav`` dan ``evil.wav``. Nunca decide dónde
    se escribe el archivo.
    """
    name = _CONTROL_CHARS.sub("", PureWindowsPath(raw or "").name).strip()
    if name in ("", ".", ".."):
        return _FALLBACK_FILENAME
    return name[:255]


class CopyrightAnalyzer:
    """Orquesta el análisis de un audio subido contra el catálogo indexado."""

    def __init__(
        self,
        catalog: Catalog,
        config: CopyrightIndexConfig,
        *,
        search_k: int = DEFAULT_SEARCH_K,
        staging_dir: str = DEFAULT_STAGING_DIR,
    ) -> None:
        if type(search_k) is not int or search_k <= 0:
            raise ValueError(f"search_k debe ser un entero positivo, no {search_k!r}")
        parts = PurePosixPath(staging_dir).parts
        if (
            not staging_dir
            or PurePosixPath(staging_dir).is_absolute()
            or PureWindowsPath(staging_dir).drive
            or PurePosixPath(staging_dir).as_posix() != staging_dir
            or any(part in (".", "..") for part in parts)
            or "\\" in staging_dir
            or "#" in staging_dir
        ):
            raise ValueError(f"staging_dir debe ser una ruta relativa canónica: {staging_dir!r}")
        self._catalog = catalog
        self._config = config
        self._search_k = search_k
        self._staging_dir = staging_dir
        self._lock = threading.RLock()

    @property
    def search_k(self) -> int:
        return self._search_k

    @property
    def staging_dir(self) -> str:
        return self._staging_dir

    def analyze(self, upload: BinaryIO, filename: str | None) -> AnalysisResult:
        """Analiza el audio de ``upload`` y devuelve la evidencia por runs.

        Lanza :class:`InvalidAudioError` si el archivo no sirve y
        :class:`AnalysisUnavailableError` si falta índice o perfil. Cualquier otro
        error es inesperado.
        """
        analysis_id = uuid.uuid4().hex
        name = display_filename(filename)
        extension = PurePosixPath(name).suffix.lower()
        if extension not in _AUDIO_EXTENSIONS:
            extension = ""
        relative = f"{self._staging_dir}/{analysis_id}/candidate{extension}"
        target = self._config.dataset_root / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=False)
            with target.open("xb") as out:
                shutil.copyfileobj(upload, out, _CHUNK_BYTES)
            if target.stat().st_size == 0:
                raise InvalidAudioError("el archivo subido está vacío")
            evidence = self._search(relative)
            return _to_analysis_result(analysis_id, name, evidence)
        finally:
            target.unlink(missing_ok=True)
            try:
                target.parent.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:  # no debería quedar nada: solo se escribe target
                logger.warning("no se pudo borrar el staging %s: %s", target.parent, exc)

    def _search(self, relative: str) -> TemporalEvidenceResult:
        with self._lock:
            try:
                search_result = search_audio(
                    self._catalog,
                    self._config,
                    relative,
                    k=self._search_k,
                )
            # Mensajes para el cliente sin rutas internas; el detalle queda en la causa.
            except CandidateAudioError as exc:
                logger.info("audio candidato rechazado: %s", exc)
                raise InvalidAudioError(
                    "el audio no se puede decodificar o es demasiado corto "
                    f"(se necesita al menos un segmento de {SEGMENT_SECONDS / 2:g} s)"
                ) from exc
            except SearchUnavailableError as exc:
                logger.error("índice de copyright no disponible: %s", exc)
                raise AnalysisUnavailableError("el índice de copyright no está disponible") from exc
            evidence = build_temporal_evidence(search_result)
        return evidence


def _to_analysis_result(
    analysis_id: str, filename: str, evidence: TemporalEvidenceResult
) -> AnalysisResult:
    """Una ``CopyrightMatch`` por ``TemporalRun``, en el orden de la evidencia."""
    matches = [
        CopyrightMatch(
            query_start=run.query_start_seconds,
            query_end=run.query_end_seconds,
            song_id=run.song_id,
            title=run.title,
            artist=run.artist,
            source_start=run.source_start_seconds,
            source_end=run.source_end_seconds,
            score=run.mean_score,
            segment_count=run.segment_count,
            min_score=run.min_score,
            max_score=run.max_score,
            alignment_offset_seconds=run.alignment_offset_seconds,
            # Identidad, no otro criterio: la misma run que eligió build_temporal_evidence.
            is_best_run=run is song.best_run,
        )
        for song in evidence.songs
        for run in song.runs
    ]
    return AnalysisResult(
        analysis_id=analysis_id,
        filename=filename,
        segment_seconds=SEGMENT_SECONDS,
        matches=matches,
    )
