"""Ingesta de canciones (issue #174) con WAV reales y el Catalog real de SoupDB."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from soupdb_vector.multimedia.audio_features import (
    MIN_SEGMENT_RATIO,
    SAMPLE_RATE,
    segment_id,
)
from soupdb_vector.multimedia.audio_features import SEGMENT_SECONDS as PIPELINE_SEGMENT_SECONDS

from applications.copyright_detection.catalog import (
    AudioSegment,
    CopyrightCatalog,
    Song,
)
from applications.copyright_detection.config import SEGMENT_SECONDS
from applications.copyright_detection.ingestion import (
    AudioIngestionError,
    SongMetadata,
    ingest_song,
    resolve_dataset_path,
)
from engine.common.catalog import Catalog

SEGMENT_SAMPLES = round(SEGMENT_SECONDS * SAMPLE_RATE)


def write_wav(path: Path, samples: int, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(samples) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, np.stack([tone] * channels, axis=1), sample_rate)
    return path


def write_seconds(root: Path, rel: str, seconds: float, **kwargs) -> Path:
    rate = kwargs.get("sample_rate", SAMPLE_RATE)
    return write_wav(root / rel, round(seconds * rate), **kwargs)


def meta(song_id=1, path="tracks/song.wav", title="Canción", artist="Artista"):
    return SongMetadata(song_id, title, artist, path)


@contextmanager
def opened(db_path: Path) -> Iterator[CopyrightCatalog]:
    catalog = Catalog(db_path)
    try:
        copyright_catalog = CopyrightCatalog(catalog)
        copyright_catalog.initialize()
        yield copyright_catalog
    finally:
        catalog.close()


@pytest.fixture
def root(tmp_path) -> Path:
    folder = tmp_path / "dataset"
    folder.mkdir()
    return folder


@pytest.fixture
def db(tmp_path) -> Iterator[CopyrightCatalog]:
    with opened(tmp_path / "db") as copyright_catalog:
        yield copyright_catalog


def test_app_segment_seconds_matches_pipeline():
    assert SEGMENT_SECONDS == PIPELINE_SEGMENT_SECONDS


# --- ingesta normal ---------------------------------------------------------


def test_ingest_13s_track_registers_song_and_three_segments(db, root):
    write_seconds(root, "tracks/song.wav", 13.0)

    result = ingest_song(db, root, meta(), first_segment_id=100)

    song = db.get_song(1)
    assert song == result.song
    assert song == Song(1, "Canción", "Artista", "tracks/song.wav", 13.0)
    segments = db.list_segments(1)
    assert segments == list(result.segments)
    assert [s.segment_index for s in segments] == [0, 1, 2]
    assert [s.segment_id for s in segments] == [100, 101, 102]
    assert [(s.start_seconds, s.end_seconds) for s in segments] == [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 13.0),
    ]
    assert [s.audio_ref for s in segments] == [
        "tracks/song.wav#segment=000000",
        "tracks/song.wav#segment=000001",
        "tracks/song.wav#segment=000002",
    ]


def test_duration_uses_pipeline_decoding_for_stereo_44k(db, root):
    write_seconds(root, "tracks/stereo.wav", 13.0, sample_rate=44100, channels=2)

    result = ingest_song(db, root, meta(path="tracks/stereo.wav"), first_segment_id=1)

    assert result.song.duration_seconds == pytest.approx(13.0, abs=1e-3)
    assert len(result.segments) == 3
    assert result.segments[-1].end_seconds == pytest.approx(13.0, abs=1e-3)


def test_duration_is_not_rounded(db, root):
    samples = 2 * SEGMENT_SAMPLES + 12345
    write_wav(root / "tracks/odd.wav", samples)

    result = ingest_song(db, root, meta(path="tracks/odd.wav"), first_segment_id=1)

    assert result.song.duration_seconds == samples / SAMPLE_RATE
    assert db.get_song(1).duration_seconds == samples / SAMPLE_RATE


# --- cola corta -------------------------------------------------------------


def test_12s_track_keeps_full_duration_but_drops_short_tail(db, root):
    write_seconds(root, "tracks/song.wav", 12.0)

    result = ingest_song(db, root, meta(), first_segment_id=1)

    assert db.get_song(1).duration_seconds == 12.0
    segments = db.list_segments(1)
    assert [(s.start_seconds, s.end_seconds) for s in segments] == [(0.0, 5.0), (5.0, 10.0)]
    assert len(result.segments) == 2
    assert db.get_segment(3) is None  # no hay fila para 10–12 s


@pytest.mark.parametrize(("tail_offset", "kept"), [(-1, False), (0, True), (1, True)])
def test_last_segment_threshold_follows_pipeline(db, root, tail_offset, kept):
    # Cola alrededor de MIN_SEGMENT_RATIO de un segmento; la decisión la toma
    # iter_audio_segments, el test solo observa cuántas filas quedan.
    tail = round(MIN_SEGMENT_RATIO * SEGMENT_SAMPLES) + tail_offset
    write_wav(root / "tracks/edge.wav", 2 * SEGMENT_SAMPLES + tail)

    result = ingest_song(db, root, meta(path="tracks/edge.wav"), first_segment_id=1)

    assert len(db.list_segments(1)) == (3 if kept else 2)
    assert len(result.segments) == (3 if kept else 2)
    if kept:
        assert result.segments[-1].end_seconds == 10.0 + tail / SAMPLE_RATE


# --- sin segmentos / audio inválido ------------------------------------------------


@pytest.mark.parametrize("seconds", [0.0, 2.0])
def test_audio_without_segments_is_rejected_and_nothing_is_written(db, root, seconds):
    write_seconds(root, "tracks/short.wav", seconds)

    with pytest.raises(AudioIngestionError, match="ningún segmento"):
        ingest_song(db, root, meta(path="tracks/short.wav"), first_segment_id=1)

    assert db.get_song(1) is None
    assert db.list_songs() == []
    assert db.get_segment(1) is None


def test_undecodable_audio_is_rejected_with_cause(db, root):
    bad = root / "tracks/bad.wav"
    bad.parent.mkdir()
    bad.write_bytes(b"esto no es audio" * 64)

    with pytest.raises(AudioIngestionError, match="no se pudo decodificar") as info:
        ingest_song(db, root, meta(path="tracks/bad.wav"), first_segment_id=1)

    assert info.value.__cause__ is not None
    assert db.list_songs() == []


# --- rutas ------------------------------------------------------------------


def test_valid_nested_path_resolves_inside_root(root):
    wav = write_seconds(root, "a/b/c.wav", 6.0)

    assert resolve_dataset_path(root, "a/b/c.wav") == wav.resolve()


def test_missing_file_is_rejected(db, root):
    with pytest.raises(AudioIngestionError, match="no existe"):
        ingest_song(db, root, meta(path="tracks/missing.wav"), first_segment_id=1)
    assert db.list_songs() == []


def test_absolute_path_is_rejected(db, root):
    wav = write_seconds(root, "tracks/song.wav", 6.0)

    with pytest.raises(AudioIngestionError, match="relativo"):
        ingest_song(db, root, meta(path=wav.resolve().as_posix()), first_segment_id=1)
    assert db.list_songs() == []


@pytest.mark.parametrize("path", ["/tracks/song.wav", "C:/tracks/song.wav", "C:song.wav"])
def test_absolute_like_paths_are_rejected(root, path):
    with pytest.raises(AudioIngestionError, match="relativo"):
        resolve_dataset_path(root, path)


def test_traversal_outside_root_is_rejected(db, root):
    write_seconds(root.parent, "outside.wav", 6.0)

    with pytest.raises(AudioIngestionError, match="sale de la raíz"):
        ingest_song(db, root, meta(path="../outside.wav"), first_segment_id=1)
    assert db.list_songs() == []


def test_directory_is_rejected(root):
    (root / "tracks").mkdir()

    with pytest.raises(AudioIngestionError, match="no es un archivo"):
        resolve_dataset_path(root, "tracks")


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ("tracks/song.wav#segment=000001", "'#'"),
        ("tracks\\song.wav", "'/'"),
        ("tracks/./song.wav", "canónica"),
        ("tracks//song.wav", "canónica"),
        ("tracks/../tracks/song.wav", "canónica"),
    ],
)
def test_ambiguous_paths_are_rejected(root, path, match):
    write_seconds(root, "tracks/song.wav", 6.0)

    with pytest.raises(AudioIngestionError, match=match):
        resolve_dataset_path(root, path)


def test_missing_root_is_rejected(tmp_path):
    with pytest.raises(AudioIngestionError, match="raíz"):
        resolve_dataset_path(tmp_path / "no-existe", "a.wav")


# --- integridad -------------------------------------------------------------


def test_existing_song_id_is_rejected_without_writing(db, root):
    write_seconds(root, "tracks/a.wav", 6.0)
    write_seconds(root, "tracks/b.wav", 6.0)
    ingest_song(db, root, meta(song_id=1, path="tracks/a.wav"), first_segment_id=1)

    with pytest.raises(AudioIngestionError, match="ya está registrada"):
        ingest_song(db, root, meta(song_id=1, path="tracks/b.wav"), first_segment_id=50)

    assert db.get_song(1).audio_path == "tracks/a.wav"
    assert db.get_segment(50) is None


def test_same_audio_path_twice_is_rejected(db, root):
    write_seconds(root, "tracks/a.wav", 6.0)
    ingest_song(db, root, meta(song_id=1, path="tracks/a.wav"), first_segment_id=1)

    with pytest.raises(AudioIngestionError, match="ya está registrado como song_id=1"):
        ingest_song(db, root, meta(song_id=2, path="tracks/a.wav"), first_segment_id=50)
    assert db.get_song(2) is None


def test_segment_id_collision_is_detected_before_writing(db, root):
    write_seconds(root, "tracks/a.wav", 13.0)
    write_seconds(root, "tracks/b.wav", 13.0)
    ingest_song(db, root, meta(song_id=1, path="tracks/a.wav"), first_segment_id=100)

    # b necesita 98, 99, 100: el 100 ya es de a.
    with pytest.raises(AudioIngestionError, match=r"segment_id \[100\] ya existen"):
        ingest_song(db, root, meta(song_id=2, path="tracks/b.wav"), first_segment_id=98)

    assert db.get_song(2) is None
    assert db.get_segment(98) is None
    assert db.get_segment(99) is None
    assert [s.song_id for s in db.list_songs()] == [1]
    assert [s.segment_id for s in db.list_segments(1)] == [100, 101, 102]


def test_orphan_segments_for_song_id_are_detected(tmp_path, root):
    # No se pueden crear con CopyrightCatalog; se simulan con SQL directo.
    from engine.query import execute_sql

    write_seconds(root, "tracks/a.wav", 6.0)
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        execute_sql(
            "INSERT INTO audio_segments VALUES (7, 1, 0, 0.0, 5.0, 'tracks/a.wav#segment=000000')",
            catalog,
        )

        with pytest.raises(AudioIngestionError, match="ya hay segmentos"):
            ingest_song(db, root, meta(path="tracks/a.wav"), first_segment_id=1)
        assert db.get_song(1) is None
    finally:
        catalog.close()


def test_segment_id_overflow_is_detected_before_writing(db, root):
    write_seconds(root, "tracks/a.wav", 13.0)

    with pytest.raises(AudioIngestionError, match="máximo de INT"):
        ingest_song(db, root, meta(path="tracks/a.wav"), first_segment_id=2**31 - 2)
    assert db.list_songs() == []


@pytest.mark.parametrize("first", [0, -1, True, 1.5])
def test_first_segment_id_must_be_positive_int(db, root, first):
    write_seconds(root, "tracks/a.wav", 6.0)

    with pytest.raises(ValueError, match="first_segment_id"):
        ingest_song(db, root, meta(path="tracks/a.wav"), first_segment_id=first)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [({"song_id": 0}, "song_id"), ({"title": " "}, "title"), ({"artist": ""}, "artist")],
)
def test_metadata_is_validated(kwargs, field):
    with pytest.raises(ValueError, match=field):
        meta(**kwargs)


# --- persistencia -----------------------------------------------------------


def test_ingested_song_persists_after_reopen(tmp_path, root):
    write_seconds(root, "tracks/song.wav", 13.0)
    with opened(tmp_path / "db") as db:
        result = ingest_song(db, root, meta(), first_segment_id=10)

    with opened(tmp_path / "db") as db:
        assert db.get_song(1) == result.song
        assert db.list_segments(1) == list(result.segments)
        assert db.get_segment(12).end_seconds == 13.0


# --- compatibilidad multimedia --------------------------------------------------


def test_audio_refs_match_soupdb_vector_segment_id(db, root):
    write_seconds(root, "rock/a.wav", 13.0)

    result = ingest_song(db, root, meta(path="rock/a.wav"), first_segment_id=1)

    assert [s.audio_ref for s in result.segments] == [
        segment_id("rock/a.wav", s.segment_index) for s in result.segments
    ]
    assert all(isinstance(s, AudioSegment) for s in result.segments)
