"""Evidencia temporal: agrupa los vecinos por segmento de #178 en secuencias por canción.

Transformación pura de un ``AudioSearchResult``: no abre archivos, no ejecuta SQL,
no conoce HNSW ni FastAPI y no vuelve a buscar.

Para cada match, ``alignment_offset_segments = source_segment_index -
query_segment_index``. Los matches de una misma canción con el mismo offset y
segmentos de consulta consecutivos forman una ``TemporalRun``: el candidato
reproduce un tramo continuo del original con ese desplazamiento.

Esto es **evidencia de similitud, no una decisión**. No hay umbral: se usan todos
los matches recibidos, también los de score bajo; las métricas de cada run
(``segment_count``, ``mean_score``, ``min_score``, ``max_score``) permiten que una
capa posterior decida qué es significativo. ``score`` sigue siendo la similitud
del motor, no una probabilidad de copyright.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from applications.copyright_detection.search import AudioSearchResult


class CopyrightEvidenceError(Exception):
    """El resultado de búsqueda es estructuralmente inconsistente."""


@dataclass(frozen=True, slots=True)
class EvidencePoint:
    """Un match: segmento del candidato frente a segmento original, sin redondear."""

    query_segment_index: int
    query_start_seconds: float
    query_end_seconds: float
    segment_id: int
    song_id: int
    source_segment_index: int
    source_start_seconds: float
    source_end_seconds: float
    score: float

    @property
    def alignment_offset_segments(self) -> int:
        return self.source_segment_index - self.query_segment_index


@dataclass(frozen=True, slots=True)
class TemporalRun:
    """Secuencia de puntos de una canción con offset constante y consultas consecutivas."""

    song_id: int
    title: str
    artist: str
    alignment_offset_segments: int
    #: ``source_start_seconds - query_start_seconds`` del primer punto.
    alignment_offset_seconds: float
    query_start_seconds: float
    query_end_seconds: float
    source_start_seconds: float
    source_end_seconds: float
    segment_count: int
    mean_score: float
    min_score: float
    max_score: float
    points: tuple[EvidencePoint, ...]


@dataclass(frozen=True, slots=True)
class SongEvidence:
    """Runs de una canción en orden temporal del candidato, y la mejor de ellas."""

    song_id: int
    title: str
    artist: str
    runs: tuple[TemporalRun, ...]
    best_run: TemporalRun


@dataclass(frozen=True, slots=True)
class TemporalEvidenceResult:
    """Evidencia por canción, ordenada de más a menos coherente temporalmente.

    El orden no expresa una probabilidad de infracción.
    """

    audio_path: str
    k: int
    songs: tuple[SongEvidence, ...]


def build_temporal_evidence(search_result: AudioSearchResult) -> TemporalEvidenceResult:
    """Agrupa los matches por canción y offset y los corta en runs consecutivas."""
    metadata: dict[int, tuple[str, str]] = {}
    groups: dict[tuple[int, int], dict[int, EvidencePoint]] = defaultdict(dict)

    for query_segment in search_result.segments:
        for match in query_segment.matches:
            known = metadata.setdefault(match.song_id, (match.title, match.artist))
            if known != (match.title, match.artist):
                raise CopyrightEvidenceError(
                    f"song_id={match.song_id} aparece con metadatos distintos: "
                    f"{known!r} y {(match.title, match.artist)!r}"
                )
            point = EvidencePoint(
                query_segment_index=query_segment.query_segment_index,
                query_start_seconds=query_segment.query_start_seconds,
                query_end_seconds=query_segment.query_end_seconds,
                segment_id=match.segment_id,
                song_id=match.song_id,
                source_segment_index=match.source_segment_index,
                source_start_seconds=match.source_start_seconds,
                source_end_seconds=match.source_end_seconds,
                score=match.score,
            )
            slot = groups[(point.song_id, point.alignment_offset_segments)]
            if point.query_segment_index in slot:
                raise CopyrightEvidenceError(
                    f"match duplicado: song_id={point.song_id}, "
                    f"offset={point.alignment_offset_segments}, "
                    f"query_segment_index={point.query_segment_index}"
                )
            slot[point.query_segment_index] = point

    runs_by_song: dict[int, list[TemporalRun]] = defaultdict(list)
    for (song_id, offset), slot in groups.items():
        title, artist = metadata[song_id]
        ordered = [slot[index] for index in sorted(slot)]
        for points in _consecutive(ordered):
            runs_by_song[song_id].append(_run(song_id, title, artist, offset, points))

    songs = []
    for song_id, runs in runs_by_song.items():
        runs.sort(
            key=lambda r: (
                r.query_start_seconds,
                r.source_start_seconds,
                r.alignment_offset_segments,
            )
        )
        title, artist = metadata[song_id]
        best = min(runs, key=_best_run_key)
        songs.append(SongEvidence(song_id, title, artist, tuple(runs), best))
    songs.sort(key=lambda s: (*_strength_key(s.best_run), s.song_id))

    return TemporalEvidenceResult(
        audio_path=search_result.audio_path, k=search_result.k, songs=tuple(songs)
    )


def _consecutive(points: list[EvidencePoint]) -> list[list[EvidencePoint]]:
    """Corta en runs donde los índices de consulta (y de origen) dejan de ser consecutivos."""
    runs: list[list[EvidencePoint]] = []
    for point in points:
        if runs:
            previous = runs[-1][-1]
            if (
                point.query_segment_index == previous.query_segment_index + 1
                and point.source_segment_index == previous.source_segment_index + 1
            ):
                runs[-1].append(point)
                continue
        runs.append([point])
    return runs


def _run(
    song_id: int, title: str, artist: str, offset: int, points: list[EvidencePoint]
) -> TemporalRun:
    first, last = points[0], points[-1]
    scores = [point.score for point in points]
    return TemporalRun(
        song_id=song_id,
        title=title,
        artist=artist,
        alignment_offset_segments=offset,
        alignment_offset_seconds=first.source_start_seconds - first.query_start_seconds,
        query_start_seconds=first.query_start_seconds,
        query_end_seconds=last.query_end_seconds,
        source_start_seconds=first.source_start_seconds,
        source_end_seconds=last.source_end_seconds,
        segment_count=len(points),
        mean_score=fmean(scores),
        min_score=min(scores),
        max_score=max(scores),
        points=tuple(points),
    )


def _strength_key(run: TemporalRun) -> tuple[int, float, float]:
    # Continuidad primero, luego score medio y mínimo (orden descendente).
    return (-run.segment_count, -run.mean_score, -run.min_score)


def _best_run_key(run: TemporalRun) -> tuple[int, float, float, float, float]:
    return (*_strength_key(run), run.query_start_seconds, run.source_start_seconds)
