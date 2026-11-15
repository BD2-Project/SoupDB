"""Aplicación FastAPI del detector de copyright/remixes de audio.

La API REST es la frontera de la aplicación: el frontend solo consume
``/api/v1/...`` y nunca accede al ``Catalog`` ni a los archivos internos.

``create_app`` solo construye la aplicación HTTP. Los servicios se leen de
``app.state`` en cada petición: pueden pasarse ya creados o instalarlos un
``lifespan``, que es lo que hace el servidor real
(:mod:`applications.copyright_detection.runtime`):

- ``analyzer``: ``POST /analyze`` (503 si falta).
- ``library``: ``GET /catalog``, ``/catalog/{song_id}`` y ``/catalog/{song_id}/audio``
  (503 si falta). El catálogo es de solo lectura: no hay endpoints de escritura.
- ``history``: ``GET /analyses``, ``/analyses/{analysis_id}`` y ``.../export``
  (503 si falta).

``/health`` responde siempre.

Importar este módulo no abre bases, no lee variables de entorno, no entrena
codebooks ni carga el pipeline multimedia: ``analysis`` solo se importa para tipos.
La ``app`` global de este módulo no tiene analizador; el servidor ejecutable es
``python -m applications.copyright_detection.runtime``.
"""

import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from applications.copyright_detection.config import API_PREFIX, SERVICE_NAME
from applications.copyright_detection.errors import AnalysisUnavailableError, InvalidAudioError
from applications.copyright_detection.export import export_filename, to_csv, to_json
from applications.copyright_detection.schemas import (
    AnalysisResult,
    AnalysisSummary,
    CatalogResponse,
    CatalogSegment,
    CatalogSong,
    CatalogSongDetail,
    ErrorResponse,
    HealthResponse,
    StoredAnalysis,
)

if TYPE_CHECKING:
    from applications.copyright_detection.analysis import CopyrightAnalyzer
    from applications.copyright_detection.history import AnalysisHistory
    from applications.copyright_detection.library import CatalogLibrary, SongEntry

logger = logging.getLogger(__name__)

ANALYZER_NOT_CONFIGURED = "El servicio de análisis no está configurado."
LIBRARY_NOT_CONFIGURED = "El catálogo no está configurado."
HISTORY_NOT_CONFIGURED = "El historial de análisis no está configurado."
SONG_NOT_FOUND = "La canción no existe en el catálogo."
AUDIO_NOT_AVAILABLE = "El audio de la canción no está disponible."
ANALYSIS_NOT_FOUND = "El análisis no existe."
INTERNAL_ERROR = "Error interno del análisis."
#: Lo necesario para ``GET``, ``POST /analyze`` multipart y peticiones ``Range`` de audio.
CORS_METHODS = ("GET", "POST")
CORS_HEADERS = ("Accept", "Content-Type", "Range")
#: Cabeceras que el navegador deja leer al frontend (descargas y audio parcial).
CORS_EXPOSE_HEADERS = ("Accept-Ranges", "Content-Disposition", "Content-Length", "Content-Range")
MAX_SONG_ID = 2**31 - 1

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
SongId = Annotated[int, PathParam(gt=0, le=MAX_SONG_ID, description="song_id del catálogo.")]
_NOT_FOUND = {status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}}
_UNAVAILABLE = {status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}}


def audio_url(song_id: int) -> str:
    return f"{API_PREFIX}/catalog/{song_id}/audio"


def _catalog_song(entry: "SongEntry") -> CatalogSong:
    song = entry.song
    return CatalogSong(
        song_id=song.song_id,
        title=song.title,
        artist=song.artist,
        duration_seconds=song.duration_seconds,
        segment_count=len(entry.segments),
        audio_url=audio_url(song.song_id),
    )


def create_app(
    analyzer: "CopyrightAnalyzer | None" = None,
    *,
    library: "CatalogLibrary | None" = None,
    history: "AnalysisHistory | None" = None,
    lifespan: Lifespan | None = None,
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    """Construye la API.

    - ``analyzer``: servicio ya configurado, o ``None`` (lo instala ``lifespan``
      en ``app.state.analyzer``, o ``/analyze`` responde 503).
    - ``library`` / ``history``: catálogo de solo lectura e historial; igual que
      ``analyzer``, pueden instalarse después en ``app.state``.
    - ``lifespan``: ciclo de vida de FastAPI (abrir/cerrar recursos).
    - ``cors_origins``: orígenes del frontend permitidos. Vacío: sin CORS.
    """
    app = FastAPI(
        lifespan=lifespan,
        title=SERVICE_NAME,
        description=(
            "API de análisis de similitud acústica y temporal sobre SoupDB. Devuelve "
            "evidencia de similitud; no determina copyright."
        ),
        version="0.2.0",
    )
    app.state.analyzer = analyzer
    app.state.library = library
    app.state.history = history
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=list(CORS_METHODS),
            allow_headers=list(CORS_HEADERS),
            expose_headers=list(CORS_EXPOSE_HEADERS),
            allow_credentials=False,
        )
    router = APIRouter(prefix=API_PREFIX)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # Sin traceback para el cliente; queda en el log del servidor.
        logger.exception("error inesperado en la API de copyright", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": INTERNAL_ERROR})

    @router.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post(
        "/analyze",
        response_model=AnalysisResult,
        tags=["analysis"],
        summary="Analizar la similitud de un audio con el catálogo",
        responses={
            422: {
                "description": (
                    "Falta el campo `file`, o el audio está vacío, no se puede "
                    "decodificar o es demasiado corto para producir segmentos."
                ),
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "Análisis no configurado o índice/perfil no disponible.",
            },
            status.HTTP_500_INTERNAL_SERVER_ERROR: {
                "model": ErrorResponse,
                "description": "Error interno inesperado.",
            },
        },
    )
    def analyze(
        request: Request,
        file: UploadFile = File(description="Archivo de audio a analizar."),  # noqa: B008
    ) -> AnalysisResult:
        analyzer: CopyrightAnalyzer | None = request.app.state.analyzer
        if analyzer is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, ANALYZER_NOT_CONFIGURED)
        try:
            return analyzer.analyze(file.file, file.filename)
        except InvalidAudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        except AnalysisUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    # --- catálogo (solo lectura) ----------------------------------------------

    def get_library(request: Request) -> "CatalogLibrary":
        library = request.app.state.library
        if library is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, LIBRARY_NOT_CONFIGURED)
        return library

    @router.get(
        "/catalog",
        response_model=CatalogResponse,
        tags=["catalog"],
        summary="Canciones originales del catálogo (solo lectura)",
        responses=_UNAVAILABLE,
    )
    def catalog(request: Request) -> CatalogResponse:
        songs = [_catalog_song(entry) for entry in get_library(request).list_songs()]
        return CatalogResponse(
            song_count=len(songs),
            segment_count=sum(song.segment_count for song in songs),
            songs=songs,
        )

    @router.get(
        "/catalog/{song_id}",
        response_model=CatalogSongDetail,
        tags=["catalog"],
        summary="Detalle de una canción y sus segmentos indexados",
        responses={**_NOT_FOUND, **_UNAVAILABLE},
    )
    def catalog_song(request: Request, song_id: SongId) -> CatalogSongDetail:
        entry = get_library(request).get_song(song_id)
        if entry is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, SONG_NOT_FOUND)
        return CatalogSongDetail(
            **_catalog_song(entry).model_dump(),
            segments=[
                CatalogSegment(
                    segment_id=s.segment_id,
                    segment_index=s.segment_index,
                    start_seconds=s.start_seconds,
                    end_seconds=s.end_seconds,
                )
                for s in entry.segments
            ],
        )

    @router.get(
        "/catalog/{song_id}/audio",
        response_class=FileResponse,
        tags=["catalog"],
        summary="Audio original de una canción catalogada (admite Range)",
        responses={
            200: {"content": {"audio/mpeg": {}, "audio/wav": {}}},
            206: {"description": "Contenido parcial (cabecera Range)."},
            **_NOT_FOUND,
            **_UNAVAILABLE,
        },
    )
    def catalog_song_audio(request: Request, song_id: SongId) -> FileResponse:
        # El archivo sale solo del song_id registrado; el cliente nunca envía rutas.
        audio = get_library(request).reference_audio(song_id)
        if audio is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, AUDIO_NOT_AVAILABLE)
        return FileResponse(
            audio.path,
            media_type=audio.media_type,
            filename=audio.filename,
            content_disposition_type="inline",
        )

    # --- historial -------------------------------------------------------------

    def get_history(request: Request) -> "AnalysisHistory":
        history = request.app.state.history
        if history is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, HISTORY_NOT_CONFIGURED)
        return history

    def get_stored(request: Request, analysis_id: str) -> StoredAnalysis:
        stored = get_history(request).get(analysis_id)
        if stored is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, ANALYSIS_NOT_FOUND)
        return stored

    @router.get(
        "/analyses",
        response_model=list[AnalysisSummary],
        tags=["history"],
        summary="Análisis guardados, del más reciente al más antiguo",
        responses=_UNAVAILABLE,
    )
    def analyses(request: Request) -> list[AnalysisSummary]:
        return get_history(request).list_summaries()

    @router.get(
        "/analyses/{analysis_id}",
        response_model=StoredAnalysis,
        tags=["history"],
        summary="Resultado completo de un análisis guardado (sin el audio subido)",
        responses={**_NOT_FOUND, **_UNAVAILABLE},
    )
    def analysis_detail(request: Request, analysis_id: str) -> StoredAnalysis:
        return get_stored(request, analysis_id)

    @router.get(
        "/analyses/{analysis_id}/export",
        tags=["history"],
        summary="Descargar un análisis guardado en JSON o CSV",
        responses={
            200: {"content": {"application/json": {}, "text/csv": {}}},
            **_NOT_FOUND,
            **_UNAVAILABLE,
        },
    )
    def analysis_export(
        request: Request,
        analysis_id: str,
        format: Annotated[Literal["json", "csv"], Query(description="json o csv")] = "json",  # noqa: A002
    ) -> Response:
        stored = get_stored(request, analysis_id)
        if format == "csv":
            body, media_type = to_csv(stored), "text/csv; charset=utf-8"
        else:
            body, media_type = to_json(stored), "application/json"
        filename = export_filename(stored.analysis_id, format)
        return Response(
            content=body,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    app.include_router(router)
    return app


app = create_app()
