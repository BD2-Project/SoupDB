"""Evidencia temporal (issue #180): transformación pura de AudioSearchResult.

Los tests puros construyen los resultados a mano. Un único test de integración
encadena ``search_audio`` (HNSW real) con ``build_temporal_evidence``.
"""

import random
from statistics import fmean

import pytest

from applications.copyright_detection.evidence import (
    CopyrightEvidenceError,
    EvidencePoint,
    SongEvidence,
    TemporalEvidenceResult,
    TemporalRun,
    build_temporal_evidence,
)
from applications.copyright_detection.search import (
    AudioSearchResult,
    QuerySegmentResult,
    SegmentMatch,
)

TITLES = {1: ("Alpha", "Ana"), 2: ("Beta", "Bruno"), 3: ("Gamma", "Gina")}


def match(song_id, source, score, *, start=None, end=None, title=None):
    start = source * 5.0 if start is None else start
    end = start + 5.0 if end is None else end
    default_title, artist = TITLES[song_id]
    return SegmentMatch(
        segment_id=song_id * 1000 + source,
        song_id=song_id,
        title=title or default_title,
        artist=artist,
        source_segment_index=source,
        source_start_seconds=start,
        source_end_seconds=end,
        audio_ref=f"tracks/{song_id}.wav#segment={source:06d}",
        score=score,
    )


def query(index, *matches, end=None):
    start = index * 5.0
    return QuerySegmentResult(index, start, start + 5.0 if end is None else end, tuple(matches))


def result(*segments, k=5):
    return AudioSearchResult("queries/x.wav", k, tuple(segments))


def runs_summary(evidence: TemporalEvidenceResult):
    return [
        (
            song.song_id,
            [
                (run.alignment_offset_segments, [p.query_segment_index for p in run.points])
                for run in song.runs
            ],
        )
        for song in evidence.songs
    ]


# --- secuencias -----------------------------------------------------------------


def test_perfect_sequence_is_one_run():
    evidence = build_temporal_evidence(
        result(
            query(0, match(1, 10, 0.94)),
            query(1, match(1, 11, 0.92)),
            query(2, match(1, 12, 0.95)),
        )
    )

    assert isinstance(evidence, TemporalEvidenceResult)
    assert (evidence.audio_path, evidence.k) == ("queries/x.wav", 5)
    (song,) = evidence.songs
    assert isinstance(song, SongEvidence)
    assert (song.song_id, song.title, song.artist) == (1, "Alpha", "Ana")
    (run,) = song.runs
    assert isinstance(run, TemporalRun)
    assert song.best_run == run
    assert run.alignment_offset_segments == 10
    assert run.alignment_offset_seconds == 50.0
    assert (run.query_start_seconds, run.query_end_seconds) == (0.0, 15.0)
    assert (run.source_start_seconds, run.source_end_seconds) == (50.0, 65.0)
    assert run.segment_count == 3
    assert run.mean_score == fmean([0.94, 0.92, 0.95])
    assert (run.min_score, run.max_score) == (0.92, 0.95)
    assert [p.segment_id for p in run.points] == [1010, 1011, 1012]
    assert all(isinstance(p, EvidencePoint) for p in run.points)
    assert run.points[1] == EvidencePoint(1, 5.0, 10.0, 1011, 1, 11, 55.0, 60.0, 0.92)


def test_gap_in_query_segments_splits_runs():
    evidence = build_temporal_evidence(
        result(
            query(0, match(1, 10, 0.9)),
            query(1, match(1, 11, 0.9)),
            query(2),
            query(3, match(1, 13, 0.9)),
        )
    )

    assert runs_summary(evidence) == [(1, [(10, [0, 1]), (10, [3])])]


def test_different_offset_breaks_the_sequence():
    evidence = build_temporal_evidence(
        result(
            query(0, match(1, 10, 0.9)),
            query(1, match(1, 18, 0.9)),
            query(2, match(1, 12, 0.9)),
        )
    )

    assert runs_summary(evidence) == [(1, [(10, [0]), (17, [1]), (10, [2])])]
    assert all(run.segment_count == 1 for run in evidence.songs[0].runs)


def test_same_song_with_two_alignments_keeps_separate_runs():
    # El estribillo (origen 2–3) aparece dos veces en el candidato.
    evidence = build_temporal_evidence(
        result(
            query(0, match(1, 2, 0.9)),
            query(1, match(1, 3, 0.9)),
            query(5, match(1, 2, 0.8)),
            query(6, match(1, 3, 0.8)),
        )
    )

    (song,) = evidence.songs
    assert [(r.alignment_offset_segments, r.segment_count) for r in song.runs] == [(2, 2), (-3, 2)]
    assert song.best_run.alignment_offset_segments == 2  # mismo largo, mayor media


def test_single_segment_run():
    (song,) = build_temporal_evidence(result(query(4, match(2, 7, 0.5)))).songs

    run = song.best_run
    assert run.segment_count == 1
    assert run.mean_score == run.min_score == run.max_score == 0.5
    assert (run.query_start_seconds, run.query_end_seconds) == (20.0, 25.0)
    assert (run.source_start_seconds, run.source_end_seconds) == (35.0, 40.0)


def test_short_final_segments_keep_real_ends():
    evidence = build_temporal_evidence(
        result(
            query(1, match(1, 5, 0.9)),
            query(2, match(1, 6, 0.8, start=30.0, end=33.0), end=13.0),
        )
    )

    run = evidence.songs[0].best_run
    assert (run.query_start_seconds, run.query_end_seconds) == (5.0, 13.0)
    assert (run.source_start_seconds, run.source_end_seconds) == (25.0, 33.0)


def test_low_scores_are_kept_without_threshold():
    evidence = build_temporal_evidence(
        result(query(0, match(1, 0, 0.01)), query(1, match(1, 1, 0.02)))
    )

    assert evidence.songs[0].best_run.segment_count == 2
    assert evidence.songs[0].best_run.max_score == 0.02


# --- varias canciones y orden -------------------------------------------------------


def test_songs_are_grouped_and_ordered_by_best_run():
    evidence = build_temporal_evidence(
        result(
            query(0, match(1, 10, 0.95), match(2, 0, 0.97), match(3, 4, 0.60)),
            query(1, match(1, 11, 0.94), match(2, 9, 0.96), match(3, 5, 0.61)),
            query(2, match(1, 12, 0.93), match(3, 6, 0.62)),
        )
    )

    assert [song.song_id for song in evidence.songs] == [1, 3, 2]
    assert [song.best_run.segment_count for song in evidence.songs] == [3, 3, 1]
    assert [song.title for song in evidence.songs] == ["Alpha", "Gamma", "Beta"]


def test_continuity_beats_a_single_high_score():
    (song,) = build_temporal_evidence(
        result(
            query(0, match(1, 0, 0.80), match(1, 20, 0.99)),
            query(1, match(1, 1, 0.80)),
            query(2, match(1, 2, 0.80)),
        )
    ).songs

    assert song.best_run.segment_count == 3
    assert song.best_run.mean_score == pytest.approx(0.80)
    assert len(song.runs) == 2


def test_best_run_tie_breaks_follow_the_documented_order():
    # Todas las runs de la canción 1 empatan en segment_count=2; se desempata
    # por mean_score, luego min_score, luego query_start y por último source_start.
    def best(*segments):
        return build_temporal_evidence(result(*segments)).songs[0].best_run

    # mean_score: 0.9 > 0.85
    assert (
        best(
            query(0, match(1, 0, 0.9)),
            query(1, match(1, 1, 0.9)),
            query(4, match(1, 10, 0.85)),
            query(5, match(1, 11, 0.85)),
        ).source_start_seconds
        == 0.0
    )
    # min_score con igual media (valores exactos en binario): (0.5, 0.5) gana a (0.25, 0.75)
    assert (
        best(
            query(0, match(1, 0, 0.25)),
            query(1, match(1, 1, 0.75)),
            query(4, match(1, 10, 0.5)),
            query(5, match(1, 11, 0.5)),
        ).source_start_seconds
        == 50.0
    )
    # query_start con métricas iguales: la que empieza antes en el candidato
    assert (
        best(
            query(0, match(1, 10, 0.8)),
            query(1, match(1, 11, 0.8)),
            query(4, match(1, 0, 0.8)),
            query(5, match(1, 1, 0.8)),
        ).query_start_seconds
        == 0.0
    )
    # source_start con igual query_start: dos alineamientos del mismo tramo
    run = best(
        query(0, match(1, 7, 0.8), match(1, 3, 0.8)),
        query(1, match(1, 8, 0.8), match(1, 4, 0.8)),
    )
    assert (run.query_start_seconds, run.source_start_seconds) == (0.0, 15.0)


def test_segment_count_dominates_song_order_then_mean_then_min_then_id():
    evidence = build_temporal_evidence(
        result(
            query(0, match(3, 0, 0.9), match(2, 0, 0.9), match(1, 0, 0.99)),
            query(1, match(3, 1, 0.9), match(2, 1, 0.9)),
        )
    )

    # 2 y 3 empatan (2 segmentos, misma media y mínimo) → menor song_id primero.
    assert [song.song_id for song in evidence.songs] == [2, 3, 1]


def test_runs_inside_a_song_are_in_candidate_time_order():
    (song,) = build_temporal_evidence(
        result(
            query(0, match(1, 9, 0.5)),
            query(2, match(1, 1, 0.9), match(1, 7, 0.4)),
            query(3, match(1, 2, 0.9)),
        )
    ).songs

    assert [(r.query_start_seconds, r.source_start_seconds) for r in song.runs] == [
        (0.0, 45.0),
        (10.0, 5.0),
        (10.0, 35.0),
    ]


# --- vacíos -----------------------------------------------------------------------


def test_no_segments_gives_empty_evidence():
    evidence = build_temporal_evidence(result())

    assert evidence == TemporalEvidenceResult("queries/x.wav", 5, ())


def test_segments_without_matches_give_empty_evidence():
    assert build_temporal_evidence(result(query(0), query(1))).songs == ()


# --- consistencia ------------------------------------------------------------------


def test_inconsistent_song_metadata_is_rejected():
    with pytest.raises(CopyrightEvidenceError, match="song_id=1"):
        build_temporal_evidence(
            result(query(0, match(1, 0, 0.9)), query(1, match(1, 1, 0.9, title="Otro")))
        )


def test_duplicate_logical_point_is_rejected():
    duplicated = match(1, 3, 0.7)
    with pytest.raises(CopyrightEvidenceError, match="duplicado"):
        build_temporal_evidence(result(query(0, match(1, 3, 0.9), duplicated)))


# --- determinismo -----------------------------------------------------------------


def test_result_does_not_depend_on_input_order():
    segments = [
        query(0, match(1, 10, 0.94), match(2, 3, 0.5), match(3, 1, 0.7)),
        query(1, match(1, 11, 0.92), match(2, 4, 0.5), match(3, 9, 0.7)),
        query(2, match(1, 12, 0.95), match(3, 3, 0.7)),
        query(3, match(2, 6, 0.5)),
    ]
    reference = build_temporal_evidence(result(*segments))

    rng = random.Random(0)
    for _ in range(10):
        shuffled = [
            QuerySegmentResult(
                s.query_segment_index,
                s.query_start_seconds,
                s.query_end_seconds,
                tuple(rng.sample(s.matches, len(s.matches))),
            )
            for s in rng.sample(segments, len(segments))
        ]
        assert build_temporal_evidence(result(*shuffled)) == reference


# --- integración search_audio → build_temporal_evidence ----------------------------


def test_real_search_result_produces_a_coherent_run(tmp_path):
    from applications.copyright_detection.catalog import CopyrightCatalog
    from applications.copyright_detection.indexing import (
        CopyrightIndexConfig,
        ensure_index,
        open_catalog,
        train_codebook,
    )
    from applications.copyright_detection.ingestion import SongMetadata, ingest_song
    from applications.copyright_detection.search import search_audio
    from engine.common.catalog import Catalog
    from tests.unit.applications.test_copyright_search import write_track

    root = tmp_path / "dataset"
    write_track(root / "tracks/a.wav", 13.0, seed=1)
    write_track(root / "tracks/b.wav", 11.0, seed=2)
    write_track(root / "queries/copy_a.wav", 13.0, seed=1)  # idéntico a tracks/a.wav
    config = CopyrightIndexConfig(
        root, tmp_path / "codebook.npz", tmp_path / "cache", tmp_path / "vectors"
    )
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        ingest_song(db, root, SongMetadata(1, "Alpha", "Ana", "tracks/a.wav"), 1)
        ingest_song(db, root, SongMetadata(2, "Beta", "Bruno", "tracks/b.wav"), 101)
        train_codebook(db, config, tmp_path / "train.jsonl")
    finally:
        catalog.close()

    catalog = open_catalog(tmp_path / "db", config)
    try:
        ensure_index(catalog, config)
        evidence = build_temporal_evidence(search_audio(catalog, config, "queries/copy_a.wav", 2))
    finally:
        catalog.close()

    top = evidence.songs[0]
    assert (top.song_id, top.title, top.artist) == (1, "Alpha", "Ana")
    run = top.best_run
    assert (run.alignment_offset_segments, run.segment_count) == (0, 3)
    assert (run.query_start_seconds, run.query_end_seconds) == (0.0, 13.0)
    assert (run.source_start_seconds, run.source_end_seconds) == (0.0, 13.0)
    assert [p.segment_id for p in run.points] == [1, 2, 3]
    assert run.min_score <= run.mean_score <= run.max_score
