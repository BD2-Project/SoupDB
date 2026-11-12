"""Catálogo relacional de copyright (issue #172) sobre el Catalog real de SoupDB."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from applications.copyright_detection.catalog import (
    AUDIO_SEGMENTS,
    AUDIO_SEGMENTS_SCHEMA,
    SONGS,
    SONGS_SCHEMA,
    AudioSegment,
    CopyrightCatalog,
    IncompatibleSchemaError,
    IntegrityError,
    Song,
    segment_audio_ref,
)
from engine.common.catalog import Catalog
from engine.query import execute_sql

SONG = Song(1, "Canción", "Artista", "tracks/song001.mp3", 183.5)
OTHER = Song(2, "Otra", "Otro artista", "tracks/song002.mp3", 60.0)


def segment(segment_id=10, song_id=1, index=0, start=None, end=None, ref=None, path=None):
    start = index * 5.0 if start is None else start
    end = start + 5.0 if end is None else end
    path = path or SONG.audio_path
    ref = segment_audio_ref(path, index) if ref is None else ref
    return AudioSegment(segment_id, song_id, index, start, end, ref)


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
def db(tmp_path) -> Iterator[CopyrightCatalog]:
    with opened(tmp_path) as copyright_catalog:
        yield copyright_catalog


# --- bootstrap --------------------------------------------------------------


def test_initialize_creates_tables_with_expected_schema(tmp_path):
    catalog = Catalog(tmp_path)
    try:
        CopyrightCatalog(catalog).initialize()

        assert catalog.schema(SONGS) == SONGS_SCHEMA
        assert catalog.schema(AUDIO_SEGMENTS) == AUDIO_SEGMENTS_SCHEMA
        assert catalog.indexes_for(SONGS, "song_id")
        assert catalog.indexes_for(AUDIO_SEGMENTS, "segment_id")
        assert catalog.indexes_for(AUDIO_SEGMENTS, "song_id")
        assert CopyrightCatalog(catalog).has_vector_index() is False
    finally:
        catalog.close()


def test_initialize_twice_keeps_data(tmp_path):
    with opened(tmp_path) as db:
        db.add_song(SONG)
        db.add_segment(segment())
        db.initialize()

        assert db.get_song(1) == SONG
        assert db.list_segments(1) == [segment()]


def test_initialize_on_reopen_keeps_data(tmp_path):
    with opened(tmp_path) as db:
        db.add_song(SONG)
    with opened(tmp_path) as db:
        assert db.list_songs() == [SONG]


def test_initialize_rejects_incompatible_schema_without_touching_data(tmp_path):
    catalog = Catalog(tmp_path)
    try:
        execute_sql("CREATE TABLE songs (song_id INT, title TEXT)", catalog)
        execute_sql("INSERT INTO songs VALUES (1, 'previa')", catalog)

        with pytest.raises(IncompatibleSchemaError, match="songs"):
            CopyrightCatalog(catalog).initialize()

        assert execute_sql("SELECT * FROM songs", catalog).rows == ((1, "previa"),)
    finally:
        catalog.close()


def test_initialize_rejects_column_type_mismatch(tmp_path):
    catalog = Catalog(tmp_path)
    try:
        columns = ", ".join(
            f"{c.name} {'TEXT' if c.name == 'end_seconds' else c.type_name.value}"
            for c in AUDIO_SEGMENTS_SCHEMA
        )
        execute_sql(f"CREATE TABLE audio_segments ({columns})", catalog)

        with pytest.raises(IncompatibleSchemaError, match="audio_segments"):
            CopyrightCatalog(catalog).initialize()
    finally:
        catalog.close()


# --- canciones --------------------------------------------------------------


def test_add_and_get_song(db):
    db.add_song(SONG)

    assert db.get_song(1) == SONG
    assert db.get_song(99) is None


def test_list_songs_ordered_by_id(db):
    db.add_song(OTHER)
    db.add_song(SONG)

    assert db.list_songs() == [SONG, OTHER]


def test_songs_persist_after_reopen(tmp_path):
    with opened(tmp_path) as db:
        db.add_song(SONG)
        db.add_song(OTHER)
    with opened(tmp_path) as db:
        assert db.get_song(2) == OTHER
        assert db.list_songs() == [SONG, OTHER]


def test_duplicate_song_id_is_rejected(db):
    db.add_song(SONG)

    with pytest.raises(IntegrityError, match="song_id 1 ya existe"):
        db.add_song(Song(1, "Otro título", "X", "tracks/x.mp3", 10.0))
    assert db.list_songs() == [SONG]


@pytest.mark.parametrize("duration", [0, 0.0, -1.5, float("nan"), float("inf")])
def test_song_rejects_invalid_duration(duration):
    with pytest.raises(ValueError, match="duration_seconds"):
        Song(1, "t", "a", "tracks/a.mp3", duration)


@pytest.mark.parametrize("song_id", [0, -1, True, 1.0])
def test_song_rejects_invalid_id(song_id):
    with pytest.raises(ValueError, match="song_id"):
        Song(song_id, "t", "a", "tracks/a.mp3", 1.0)


@pytest.mark.parametrize("path", ["", "/abs/a.mp3", "C:/a.mp3", "tracks\\a.mp3", "a.mp3#x"])
def test_song_rejects_non_relative_or_ambiguous_path(path):
    with pytest.raises(ValueError, match="audio_path"):
        Song(1, "t", "a", path, 1.0)


def test_song_accepts_int_duration_as_float():
    assert Song(1, "t", "a", "a.mp3", 3).duration_seconds == 3.0


# --- segmentos --------------------------------------------------------------


def test_add_and_get_segment_keeps_all_fields(db):
    db.add_song(SONG)
    stored = segment(segment_id=7, index=3, start=15.0, end=20.0)
    db.add_segment(stored)

    loaded = db.get_segment(7)
    assert loaded == stored
    assert loaded.segment_index == 3
    assert (loaded.start_seconds, loaded.end_seconds) == (15.0, 20.0)
    assert loaded.audio_ref == "tracks/song001.mp3#segment=000003"
    assert db.get_segment(999) is None


def test_list_segments_of_a_song_ordered_by_index(db):
    db.add_song(SONG)
    db.add_song(OTHER)
    db.add_segment(segment(segment_id=12, index=2))
    db.add_segment(segment(segment_id=10, index=0))
    db.add_segment(segment(segment_id=20, song_id=2, index=0, path=OTHER.audio_path))
    db.add_segment(segment(segment_id=11, index=1))

    assert [s.segment_id for s in db.list_segments(1)] == [10, 11, 12]
    assert [s.segment_id for s in db.list_segments(2)] == [20]
    assert db.list_segments(99) == []


def test_segment_for_missing_song_is_rejected(db):
    with pytest.raises(IntegrityError, match="song_id 1 no existe"):
        db.add_segment(segment())
    assert db.get_segment(10) is None


def test_duplicate_segment_id_is_rejected(db):
    db.add_song(SONG)
    db.add_segment(segment(segment_id=10, index=0))

    with pytest.raises(IntegrityError, match="segment_id 10 ya existe"):
        db.add_segment(segment(segment_id=10, index=1))
    assert len(db.list_segments(1)) == 1


def test_repeated_segment_index_in_same_song_is_rejected(db):
    db.add_song(SONG)
    db.add_segment(segment(segment_id=10, index=0))

    with pytest.raises(IntegrityError, match="segment_index 0"):
        db.add_segment(segment(segment_id=11, index=0))
    assert [s.segment_id for s in db.list_segments(1)] == [10]


def test_same_segment_index_in_different_songs_is_allowed(db):
    db.add_song(SONG)
    db.add_song(OTHER)
    db.add_segment(segment(segment_id=10, song_id=1, index=0))
    db.add_segment(segment(segment_id=20, song_id=2, index=0, path=OTHER.audio_path))

    assert db.get_segment(20).song_id == 2


@pytest.mark.parametrize(
    ("start", "end", "field"),
    [
        (5.0, 5.0, "end_seconds"),
        (10.0, 5.0, "end_seconds"),
        (-1.0, 4.0, "start_seconds"),
        (0.0, float("nan"), "end_seconds"),
    ],
)
def test_segment_rejects_invalid_interval(start, end, field):
    with pytest.raises(ValueError, match=field):
        segment(start=start, end=end)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"segment_id": 0}, "segment_id"),
        ({"segment_id": -3}, "segment_id"),
        ({"index": -1, "ref": "tracks/song001.mp3#segment=000000"}, "segment_index"),
        ({"ref": ""}, "audio_ref"),
    ],
)
def test_segment_rejects_invalid_values(kwargs, field):
    with pytest.raises(ValueError, match=field):
        segment(**kwargs)


@pytest.mark.parametrize(
    "ref",
    [
        "tracks/song001.mp3#segment=000001",  # índice distinto
        "tracks/song001.mp3#segment=0",  # sin el relleno de 6 dígitos
        "tracks/song002.mp3#segment=000000",  # otra canción
        "tracks/song001.mp3",  # sin sufijo
    ],
)
def test_segment_with_non_canonical_audio_ref_is_rejected(db, ref):
    db.add_song(SONG)

    with pytest.raises(IntegrityError, match="audio_ref"):
        db.add_segment(segment(index=0, ref=ref))


def test_segments_persist_after_reopen(tmp_path):
    with opened(tmp_path) as db:
        db.add_song(SONG)
        for index in range(3):
            db.add_segment(segment(segment_id=100 + index, index=index))
    with opened(tmp_path) as db:
        segments = db.list_segments(1)
        assert [s.segment_index for s in segments] == [0, 1, 2]
        assert segments[2] == segment(segment_id=102, index=2)
        assert db.get_segment(101).audio_ref == "tracks/song001.mp3#segment=000001"
        # Las invariantes siguen aplicándose sobre los datos recargados.
        with pytest.raises(IntegrityError):
            db.add_segment(segment(segment_id=101, index=5))


# --- literales SQL --------------------------------------------------------------

INJECTION = "x'); DROP TABLE songs; --"


@pytest.mark.parametrize("text", ["O'Connor", INJECTION, "''", "a'b'c", "--", ";"])
def test_text_with_quotes_roundtrips_exactly(tmp_path, text):
    song = Song(1, text, text, "tracks/a.mp3", 10.0)
    with opened(tmp_path) as db:
        db.add_song(song)
        assert db.get_song(1) == song
    with opened(tmp_path) as db:
        assert db.get_song(1).title == text
        assert db.get_song(1).artist == text


def test_sql_like_text_does_not_alter_the_statement(tmp_path):
    catalog = Catalog(tmp_path)
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        db.add_song(Song(1, INJECTION, "O'Connor", "tracks/o'connor.mp3", 10.0))
        db.add_segment(segment(segment_id=10, path="tracks/o'connor.mp3"))

        assert SONGS in catalog.tables()
        assert AUDIO_SEGMENTS in catalog.tables()
        assert execute_sql("SELECT title, artist FROM songs", catalog).rows == (
            (INJECTION, "O'Connor"),
        )
        assert db.get_segment(10).audio_ref == "tracks/o'connor.mp3#segment=000000"
        assert len(db.list_songs()) == 1
    finally:
        catalog.close()


@pytest.mark.parametrize(
    "value", [0.1 + 0.2, 1e-05, 123456789.123456789, 1e20, 5.0, 183.4712345678]
)
def test_floats_roundtrip_exactly(tmp_path, value):
    with opened(tmp_path) as db:
        db.add_song(Song(1, "t", "a", "a.mp3", value))
        db.add_segment(AudioSegment(10, 1, 0, value, value * 2, "a.mp3#segment=000000"))
    with opened(tmp_path) as db:
        assert db.get_song(1).duration_seconds == value
        stored = db.get_segment(10)
        assert (stored.start_seconds, stored.end_seconds) == (value, value * 2)


@pytest.mark.parametrize("bad", ["1 OR 1 = 1", "1", 1.0, True, 0, -1, None])
def test_lookups_only_accept_positive_int_ids(db, bad):
    db.add_song(SONG)

    with pytest.raises(ValueError):
        db.get_song(bad)
    with pytest.raises(ValueError):
        db.get_segment(bad)
    with pytest.raises(ValueError):
        db.list_segments(bad)


# --- convención multimedia de la Parte 4 ---------------------------------------


@pytest.mark.parametrize("index", [0, 1, 42, 1_234_567])
def test_audio_ref_matches_soupdb_vector_segment_id(index):
    from soupdb_vector.multimedia.audio_features import segment_id

    assert segment_audio_ref("tracks/song001.mp3", index) == segment_id("tracks/song001.mp3", index)


def test_catalog_accepts_refs_built_by_soupdb_vector(db):
    from soupdb_vector.multimedia.audio_features import segment_id

    db.add_song(SONG)
    for index in range(2):
        ref = segment_id(SONG.audio_path, index)
        db.add_segment(segment(segment_id=50 + index, index=index, ref=ref))

    assert [s.audio_ref for s in db.list_segments(1)] == [
        "tracks/song001.mp3#segment=000000",
        "tracks/song001.mp3#segment=000001",
    ]
