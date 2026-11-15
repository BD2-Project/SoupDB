"""Catálogo FMA Small (``fma_catalog``) sobre fixtures locales, sin Internet.

Los fixtures imitan el formato oficial: ``tracks.csv`` con cabecera doble (más la
fila ``track_id,,,...``) y audio en ``<3 dígitos>/<6 dígitos>.mp3``. Los MP3 se
generan en runtime con libsndfile; ninguno sale del ``tmp_path``.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from soupdb_vector.multimedia.audio_features import SAMPLE_RATE

from applications.copyright_detection import fma_catalog, runtime
from applications.copyright_detection.catalog import CopyrightCatalog
from applications.copyright_detection.fma_catalog import (
    FMA_SMALL_GENRES,
    FmaCatalogError,
    FmaTrack,
    fma_audio_relative_path,
    make_query,
    prepare_fma_catalog,
    read_fma_small_tracks,
    select_fma_small_tracks,
)
from applications.copyright_detection.indexing import (
    CODEBOOK_K,
    INDEX_NAME,
    open_catalog,
    require_index,
)
from applications.copyright_detection.ingestion import AudioIngestionError
from applications.copyright_detection.settings import RuntimeSettings
from engine.common.catalog import Catalog

LEVEL0 = ["", "album", "artist", "set", "set", "track", "track", "track", "track"]
LEVEL1 = ["", "title", "name", "split", "subset", "genre_top", "license", "title", "duration"]


def write_tracks_csv(path: Path, rows: list[dict]) -> Path:
    """``tracks.csv`` con el formato de FMA: dos filas de cabecera + fila ``track_id``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LEVEL0)
        writer.writerow(LEVEL1)
        writer.writerow(["track_id"] + [""] * (len(LEVEL0) - 1))
        for row in rows:
            writer.writerow(
                [
                    row["id"],
                    row.get("album", "Album, with comma\nand newline"),
                    row.get("artist", f"Artist {row['id']}"),
                    "training",
                    row.get("subset", "small"),
                    row["genre"],
                    row.get("license", "Attribution-NonCommercial-ShareAlike 3.0"),
                    row.get("title", f"Song {row['id']}"),
                    "30",
                ]
            )
    return path


def touch_mp3(root: Path, track_id: int) -> Path:
    path = root / fma_audio_relative_path(track_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def write_mp3(root: Path, track_id: int, seconds: float) -> Path:
    """MP3 real y determinista: acordes distintos por pista y bloque de 5 s."""
    rng = np.random.default_rng(track_id)
    samples = round(seconds * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    signal = np.zeros(samples)
    block = round(5.0 * SAMPLE_RATE)
    base = 110.0 + 37.0 * (track_id % 23)
    for start in range(0, samples, block):
        piece = slice(start, start + block)
        for ratio in rng.uniform(1.0, 6.0, size=3):
            signal[piece] += 0.2 * np.sin(2 * np.pi * base * ratio * t[piece])
    signal += 0.02 * rng.standard_normal(samples)
    path = root / fma_audio_relative_path(track_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, SAMPLE_RATE, format="MP3")
    return path


def balanced_rows(per_genre: int, genres=FMA_SMALL_GENRES, start: int = 2) -> list[dict]:
    rows, track_id = [], start
    for genre in genres:
        for _ in range(per_genre):
            rows.append({"id": track_id, "genre": genre})
            track_id += 1
    return rows


#: Cualquier ruta absoluta Windows (unidad o UNC) o POSIX de varios niveles.
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\|(?<![\w.])/[^\s/'\"]+/")


def manifest_strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in manifest_strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in manifest_strings(item)]
    return []


def assert_no_absolute_paths(manifest: dict, audio_root: Path) -> None:
    variants = {str(audio_root), audio_root.as_posix(), str(audio_root).replace("\\", "\\\\")}
    for text in manifest_strings(manifest):
        assert not any(v in text for v in variants), text
        assert not ABSOLUTE_PATH.search(text), text


# --------------------------------------------------------------------------- rutas


@pytest.mark.parametrize(
    ("track_id", "expected"),
    [(2, "000/000002.mp3"), (1234, "001/001234.mp3"), (155066, "155/155066.mp3")],
)
def test_fma_audio_relative_path(track_id, expected):
    assert fma_audio_relative_path(track_id) == expected


@pytest.mark.parametrize("track_id", [0, -1, True, "2", 2.0, 2**31])
def test_fma_audio_relative_path_rejects_invalid_ids(track_id):
    with pytest.raises(ValueError):
        fma_audio_relative_path(track_id)


# --------------------------------------------------------------------------- metadata


def test_parses_double_header_and_filters_small(tmp_path):
    csv_path = write_tracks_csv(
        tmp_path / "tracks.csv",
        [
            {"id": 5, "genre": "Rock", "title": "Five", "artist": "Band, The"},
            {"id": 2, "genre": "Pop", "title": "Two"},
            {"id": 3, "genre": "Rock", "subset": "medium"},
            {"id": 4, "genre": "Folk", "subset": "large"},
        ],
    )
    tracks = read_fma_small_tracks(csv_path)
    assert tracks == [
        FmaTrack(
            2,
            "Two",
            "Artist 2",
            "Pop",
            "000/000002.mp3",
            "Attribution-NonCommercial-ShareAlike 3.0",
        ),
        FmaTrack(
            5,
            "Five",
            "Band, The",
            "Rock",
            "000/000005.mp3",
            "Attribution-NonCommercial-ShareAlike 3.0",
        ),
    ]


def test_parses_without_index_name_row_and_fills_empty_text(tmp_path):
    path = tmp_path / "tracks.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        # Nivel 0 solo en la primera columna de cada grupo: se hereda como en pandas.
        writer.writerow(["", "artist", "set", "", "track", ""])
        writer.writerow(["", "name", "split", "subset", "genre_top", "title"])
        writer.writerow(["7", "", "training", "small", "Rock", ""])
    (track,) = read_fma_small_tracks(path)
    assert (track.track_id, track.genre) == (7, "Rock")
    assert track.title == "FMA track 7"
    assert track.artist == "Unknown artist"
    assert track.license == ""


def test_empty_metadata_fails(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(FmaCatalogError, match="cabecera doble"):
        read_fma_small_tracks(empty)

    header_only = write_tracks_csv(tmp_path / "header.csv", [])
    with pytest.raises(FmaCatalogError, match="no contiene pistas"):
        read_fma_small_tracks(header_only)

    no_small = write_tracks_csv(
        tmp_path / "medium.csv", [{"id": 2, "genre": "Rock", "subset": "medium"}]
    )
    with pytest.raises(FmaCatalogError, match="no contiene pistas"):
        read_fma_small_tracks(no_small)


def test_metadata_errors_are_clear(tmp_path):
    with pytest.raises(FmaCatalogError, match="no existe"):
        read_fma_small_tracks(tmp_path / "missing.csv")

    bad_columns = tmp_path / "bad.csv"
    bad_columns.write_text(",track\n,title\n2,Song\n", encoding="utf-8")
    with pytest.raises(FmaCatalogError, match="set/subset"):
        read_fma_small_tracks(bad_columns)

    bad_id = write_tracks_csv(tmp_path / "id.csv", [{"id": "abc", "genre": "Rock"}])
    with pytest.raises(FmaCatalogError, match="track_id inválido"):
        read_fma_small_tracks(bad_id)

    repeated = write_tracks_csv(
        tmp_path / "dup.csv", [{"id": 2, "genre": "Rock"}, {"id": 2, "genre": "Pop"}]
    )
    with pytest.raises(FmaCatalogError, match="repetido"):
        read_fma_small_tracks(repeated)


# --------------------------------------------------------------------------- selección


@pytest.fixture
def selection_dataset(tmp_path):
    """3 pistas por género con MP3 vacíos (la selección solo mira que existan)."""
    audio = tmp_path / "fma_small"
    rows = balanced_rows(3)
    for row in rows:
        touch_mp3(audio, row["id"])
    return audio, write_tracks_csv(tmp_path / "tracks.csv", rows)


def test_balanced_selection_is_deterministic(selection_dataset):
    audio, csv_path = selection_dataset
    selected = select_fma_small_tracks(csv_path, audio, per_genre=2)
    assert len(selected) == 2 * len(FMA_SMALL_GENRES)
    by_genre = {}
    for track in selected:
        by_genre.setdefault(track.genre, []).append(track.track_id)
    assert list(by_genre) == list(FMA_SMALL_GENRES)
    # Las dos primeras por track_id de cada género (3 por género, desde el id 2).
    assert by_genre["Electronic"] == [2, 3]
    assert by_genre["Rock"] == [23, 24]
    assert select_fma_small_tracks(csv_path, audio, per_genre=2) == selected


def test_selection_ignores_csv_row_order(tmp_path):
    audio = tmp_path / "fma_small"
    rows = list(reversed(balanced_rows(2)))
    for row in rows:
        touch_mp3(audio, row["id"])
    csv_path = write_tracks_csv(tmp_path / "tracks.csv", rows)
    selected = select_fma_small_tracks(csv_path, audio, per_genre=1)
    assert [t.track_id for t in selected] == [2 + 2 * i for i in range(len(FMA_SMALL_GENRES))]


def test_selection_skips_missing_files(selection_dataset):
    audio, csv_path = selection_dataset
    (audio / fma_audio_relative_path(2)).unlink()  # primera de Electronic
    selected = select_fma_small_tracks(csv_path, audio, per_genre=2)
    electronic = [t.track_id for t in selected if t.genre == "Electronic"]
    assert electronic == [3, 4]
    assert all((audio / t.source_path).is_file() for t in selected)


def test_selection_fails_when_a_genre_lacks_files(selection_dataset):
    audio, csv_path = selection_dataset
    for track_id in (2, 3):
        (audio / fma_audio_relative_path(track_id)).unlink()
    with pytest.raises(FmaCatalogError, match="'Electronic' solo tiene 1"):
        select_fma_small_tracks(csv_path, audio, per_genre=2)


def test_selection_fails_when_a_genre_is_missing(tmp_path):
    audio = tmp_path / "fma_small"
    rows = balanced_rows(1, genres=FMA_SMALL_GENRES[:-1])
    for row in rows:
        touch_mp3(audio, row["id"])
    csv_path = write_tracks_csv(tmp_path / "tracks.csv", rows)
    with pytest.raises(FmaCatalogError, match="Rock"):
        select_fma_small_tracks(csv_path, audio, per_genre=1)


@pytest.mark.parametrize("per_genre", [0, -1, True, 1.5, "10"])
def test_invalid_per_genre(selection_dataset, tmp_path, per_genre):
    audio, csv_path = selection_dataset
    with pytest.raises(FmaCatalogError, match="per_genre"):
        select_fma_small_tracks(csv_path, audio, per_genre=per_genre)
    with pytest.raises(FmaCatalogError, match="per_genre"):
        prepare_fma_catalog(audio, csv_path, tmp_path / "out", per_genre)
    assert not (tmp_path / "out").exists()


def test_audio_root_must_have_fma_layout(tmp_path, selection_dataset):
    _, csv_path = selection_dataset
    (tmp_path / "plain").mkdir()
    with pytest.raises(FmaCatalogError, match="estructura de FMA"):
        select_fma_small_tracks(csv_path, tmp_path / "plain", per_genre=1)
    with pytest.raises(FmaCatalogError, match="fma_small"):
        select_fma_small_tracks(csv_path, tmp_path, per_genre=1)
    with pytest.raises(FmaCatalogError, match="no es un directorio"):
        select_fma_small_tracks(csv_path, tmp_path / "nope", per_genre=1)


def test_prepare_fails_before_writing_when_metadata_is_short(selection_dataset, tmp_path):
    audio, csv_path = selection_dataset
    out = tmp_path / "out"
    with pytest.raises(FmaCatalogError, match="solo tiene 3 pistas en la metadata"):
        prepare_fma_catalog(audio, csv_path, out, per_genre=4)
    assert not out.exists()


# --------------------------------------------------------------------------- ingesta


def test_ingestion_skips_undecodable_tracks_without_partial_rows(tmp_path):
    """Duraciones distintas, una corrupta y una ausente: ids contiguos, sin filas sueltas."""
    audio = tmp_path / "fma_small"
    genres = ("Pop", "Rock")
    write_mp3(audio, 2, 11.0)  # Pop: 2 segmentos (cola de 1 s descartada)
    corrupt = audio / fma_audio_relative_path(3)  # Pop: existe pero no se decodifica
    corrupt.write_bytes(b"not an mp3" * 100)
    write_mp3(audio, 4, 16.0)  # Pop: 3 segmentos
    # Rock: 5 no existe; 6 y 7 sí.
    write_mp3(audio, 6, 13.0)  # Rock: 3 segmentos (cola de 3 s se conserva)
    write_mp3(audio, 7, 6.0)  # Rock: 1 segmento
    rows = [{"id": i, "genre": "Pop"} for i in (2, 3, 4)] + [
        {"id": i, "genre": "Rock"} for i in (5, 6, 7)
    ]
    tracks = read_fma_small_tracks(write_tracks_csv(tmp_path / "tracks.csv", rows))
    grouped = fma_catalog.group_by_genre(tracks, genres)

    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        selected, skipped = fma_catalog._ingest_balanced(db, audio.resolve(), grouped, 2)
        assert [(t.track_id, t.song_id, t.segment_count) for t in selected] == [
            (2, 2, 2),
            (4, 4, 3),
            (6, 6, 3),
            (7, 7, 1),
        ]
        assert [(t.first_segment_id, t.last_segment_id) for t in selected] == [
            (1, 2),
            (3, 5),
            (6, 8),
            (9, 9),
        ]
        reasons = {(s.track_id, s.reason) for s in skipped}
        assert reasons == {(3, "ingestion_failed"), (5, "missing_file")}
        assert db.get_song(3) is None and db.list_segments(3) == []
        ids = sorted(seg.segment_id for s in db.list_songs() for seg in db.list_segments(s.song_id))
        assert ids == list(range(1, 10))
    finally:
        catalog.close()


LINUX_ROOT = "/tmp/pytest-of-runner/pytest-0/fma0/fma_small"
WINDOWS_ROOT = r"C:\Users\runner\AppData\Local\Temp\pytest-0\fma0\fma_small"
TRACK = "000/000002.mp3"
UNKNOWN = "File contains data in an unknown format."


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # libsndfile en Linux (el fallo de CI).
        (f"Error opening '{LINUX_ROOT}/{TRACK}': {UNKNOWN}", f"Error opening '{TRACK}': {UNKNOWN}"),
        (f"cannot read {LINUX_ROOT}/{TRACK} now", f"cannot read {TRACK} now"),
        (f"see ({LINUX_ROOT}/{TRACK}), retry.", f"see ({TRACK}), retry."),
        # libsndfile en Windows: repr con barras duplicadas, y variantes.
        (
            "Error opening " + repr(WINDOWS_ROOT + r"\000\000002.mp3") + ": bad",
            f"Error opening '{TRACK}': bad",
        ),
        (f"failed: {WINDOWS_ROOT}" + r"\000\000002.mp3", f"failed: {TRACK}"),
        (f"failed: C:/data/fma_small/{TRACK}", f"failed: {TRACK}"),
        (r"failed: 'C:\Users\BREYSI~1\Temp\fma\000\000002.mp3'", f"failed: '{TRACK}'"),
        (r'open "D:\My Music\fma small\000\000002.mp3" failed', f'open "{TRACK}" failed'),
        (r"open \\nas\share\fma\000\000002.mp3 failed", f"open {TRACK} failed"),
        # Rutas absolutas ajenas a la pista: se redactan.
        (f"cache {LINUX_ROOT}/other.bin missing", "cache <path> missing"),
        (r"temp 'C:\Windows\Temp\x y.tmp' locked", "temp '<path>' locked"),
        # Sin rutas absolutas: el mensaje no cambia.
        ("bad header in 000/000002.mp3 (MPEG Layer I/II), ratio 1/2", None),
    ],
)
def test_safe_error_detail_strips_absolute_paths(message, expected):
    detail = fma_catalog._safe_error_detail(RuntimeError(message), "000/000002.mp3")
    assert detail == "RuntimeError: " + (message if expected is None else expected)
    assert "runner" not in detail and "BREYSI" not in detail and "My Music" not in detail


def test_safe_error_detail_keeps_ingestion_error_message():
    exc = AudioIngestionError(
        f"no se pudo decodificar '000/000002.mp3': Error opening '{LINUX_ROOT}/000/000002.mp3'"
    )
    assert fma_catalog._safe_error_detail(exc, "000/000002.mp3") == (
        "no se pudo decodificar '000/000002.mp3': Error opening '000/000002.mp3'"
    )


def test_manifest_never_stores_absolute_paths_from_exceptions(tmp_path, monkeypatch):
    """Excepciones con rutas absolutas Linux y Windows no llegan a catalog_manifest.json."""
    audio = tmp_path / "fma_small"
    for track_id in (2, 3, 4):
        touch_mp3(audio, track_id)
    root = audio.resolve()
    failures = {
        2: AudioIngestionError(
            f"no se pudo decodificar '000/000002.mp3': Error opening "
            f"'{LINUX_ROOT}/000/000002.mp3': File contains data in an unknown format."
        ),
        3: OSError(
            f"[Errno 5] I/O error: {str(root / fma_audio_relative_path(3))!r} ({WINDOWS_ROOT})"
        ),
    }
    real_ingest = fma_catalog.ingest_song

    def flaky_ingest(db, dataset_root, metadata, first_segment_id):
        if metadata.song_id in failures:
            raise failures[metadata.song_id]
        write_mp3(audio, metadata.song_id, 6.0)
        return real_ingest(db, dataset_root, metadata, first_segment_id)

    monkeypatch.setattr(fma_catalog, "ingest_song", flaky_ingest)
    rows = [{"id": i, "genre": "Pop"} for i in (2, 3, 4)]
    grouped = fma_catalog.group_by_genre(
        read_fma_small_tracks(write_tracks_csv(tmp_path / "tracks.csv", rows)), ("Pop",)
    )
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        selected, skipped = fma_catalog._ingest_balanced(db, root, grouped, 1)
    finally:
        catalog.close()

    assert [t.track_id for t in selected] == [4]
    manifest = fma_catalog.build_manifest(1, ("Pop",), selected, skipped)
    text = json.dumps(manifest, ensure_ascii=False)
    for leaked in (LINUX_ROOT, "/tmp/", WINDOWS_ROOT, "runner", str(root), root.as_posix()):
        assert leaked not in text
    assert_no_absolute_paths(manifest, root)
    details = {s["track_id"]: s["detail"] for s in manifest["skipped_tracks"]}
    assert details[2] == (
        "no se pudo decodificar '000/000002.mp3': Error opening '000/000002.mp3': "
        "File contains data in an unknown format."
    )
    assert details[3] == "OSError: [Errno 5] I/O error: '000/000003.mp3' (<path>)"
    assert {s["reason"] for s in manifest["skipped_tracks"]} == {"ingestion_failed"}


def test_ingestion_fails_when_corrupt_tracks_exhaust_a_genre(tmp_path):
    audio = tmp_path / "fma_small"
    write_mp3(audio, 2, 6.0)
    (audio / fma_audio_relative_path(3)).write_bytes(b"garbage")
    rows = [{"id": 2, "genre": "Pop"}, {"id": 3, "genre": "Pop"}]
    grouped = fma_catalog.group_by_genre(
        read_fma_small_tracks(write_tracks_csv(tmp_path / "tracks.csv", rows)), ("Pop",)
    )
    catalog = Catalog(tmp_path / "db")
    try:
        db = CopyrightCatalog(catalog)
        db.initialize()
        with pytest.raises(FmaCatalogError, match="'Pop' solo tiene 1"):
            fma_catalog._ingest_balanced(db, audio.resolve(), grouped, 2)
    finally:
        catalog.close()


# --------------------------------------------------------------------------- catálogo completo

#: Duraciones por posición dentro del género: segmentos 2, 3 y una pista de reserva.
DURATIONS = (11.0, 16.0)


@pytest.fixture(scope="module")
def fma_dataset(tmp_path_factory):
    """8 géneros × 2 pistas reales + una corrupta y una ausente en Electronic."""
    base = tmp_path_factory.mktemp("fma")
    audio = base / "fma_small"
    rows, track_id = [], 2
    for genre in FMA_SMALL_GENRES:
        extra = 2 if genre == "Electronic" else 0
        for position in range(2 + extra):
            rows.append({"id": track_id, "genre": genre, "title": f"{genre} {position}"})
            track_id += 1
    for row in rows:
        write_mp3(audio, row["id"], DURATIONS[row["id"] % 2])
    # Electronic = ids 2..5: 2 corrupto, 3 ausente; quedan 4 y 5.
    (audio / fma_audio_relative_path(2)).write_bytes(b"\x00corrupt" * 64)
    (audio / fma_audio_relative_path(3)).unlink()
    return audio, write_tracks_csv(base / "fma_metadata" / "tracks.csv", rows)


@pytest.fixture(scope="module")
def prepared(fma_dataset, tmp_path_factory):
    audio, csv_path = fma_dataset
    out = tmp_path_factory.mktemp("fma_out") / "catalog"
    result = prepare_fma_catalog(audio, csv_path, out, per_genre=2)
    assert result.reused is False
    return result


def test_prepare_builds_codebook_and_hnsw(prepared):
    paths = prepared.paths
    assert paths.codebook_path.is_file()
    assert paths.training_manifest.is_file()
    assert any(paths.vector_storage.iterdir())
    assert paths.audio_root == paths.index_config().dataset_root
    catalog = open_catalog(paths.db_path, paths.index_config())
    try:
        require_index(catalog)
        db = CopyrightCatalog(catalog)
        songs = db.list_songs()
        assert len(songs) == 2 * len(FMA_SMALL_GENRES)
        # song_id = track_id y audio_path con la convención oficial de FMA.
        assert all(s.audio_path == fma_audio_relative_path(s.song_id) for s in songs)
        segment_ids = sorted(seg.segment_id for s in songs for seg in db.list_segments(s.song_id))
        assert segment_ids == list(range(1, len(segment_ids) + 1))
        assert len(segment_ids) == prepared.manifest["number_of_segments"]
    finally:
        catalog.close()


def test_manifest_contents(prepared):
    manifest = json.loads(prepared.paths.manifest.read_text(encoding="utf-8"))
    assert manifest == prepared.manifest
    assert manifest["dataset"] == "FMA Small"
    assert manifest["source"] == "Free Music Archive"
    assert manifest["number_of_songs"] == 16
    assert manifest["genres"] == list(FMA_SMALL_GENRES)
    assert manifest["per_genre"] == 2
    assert manifest["segment_seconds"] == 5.0
    assert manifest["index"]["name"] == INDEX_NAME
    assert manifest["metric"] == "euclidean"
    assert manifest["codebook_k"] == CODEBOOK_K

    selected = manifest["selected_tracks"]
    electronic = [t["track_id"] for t in selected if t["genre"] == "Electronic"]
    assert electronic == [4, 5]
    assert {(s["track_id"], s["reason"]) for s in manifest["skipped_tracks"]} == {
        (2, "ingestion_failed"),
        (3, "missing_file"),
    }
    for track in selected:
        assert track["song_id"] == track["track_id"]
        assert track["audio_path"] == fma_audio_relative_path(track["track_id"])
        assert track["segment_count"] == (2 if track["duration_seconds"] < 12 else 3)
        assert track["last_segment_id"] - track["first_segment_id"] + 1 == track["segment_count"]
        assert {"title", "artist", "genre", "duration_seconds"} <= track.keys()
    assert manifest["number_of_segments"] == sum(t["segment_count"] for t in selected)
    # Sin rutas absolutas.
    text = prepared.paths.manifest.read_text(encoding="utf-8")
    assert str(prepared.paths.output_root) not in text
    assert str(prepared.paths.audio_root) not in text
    # En el JSON las barras de Windows van escapadas: se comprueban también los valores.
    assert_no_absolute_paths(manifest, prepared.paths.audio_root)
    corrupt = next(s for s in manifest["skipped_tracks"] if s["track_id"] == 2)
    assert "000/000002.mp3" in corrupt["detail"]


def test_prepare_does_not_touch_audio_root(prepared):
    audio = prepared.paths.audio_root
    names = {p.name for p in audio.iterdir()}
    # ``queries/`` solo lo crea el runtime (staging de uploads) en el smoke test.
    assert names - {"queries"} == {"000"}


def test_prepare_reuses_complete_catalog(prepared, fma_dataset):
    audio, csv_path = fma_dataset
    again = prepare_fma_catalog(audio, csv_path, prepared.paths.output_root, per_genre=2)
    assert again.reused is True
    assert again.manifest == prepared.manifest
    with pytest.raises(FmaCatalogError, match="per_genre=2"):
        prepare_fma_catalog(audio, csv_path, prepared.paths.output_root, per_genre=1)


def test_prepare_refuses_foreign_output_root(fma_dataset, tmp_path):
    audio, csv_path = fma_dataset
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FmaCatalogError, match="no borra directorios"):
        prepare_fma_catalog(audio, csv_path, tmp_path, per_genre=1)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_make_query_copies_catalogued_track(prepared):
    paths = prepared.paths
    target = make_query(paths.audio_root, paths.output_root)
    try:
        first = prepared.manifest["selected_tracks"][0]
        assert target == paths.sample_query
        assert target.read_bytes() == (paths.audio_root / first["audio_path"]).read_bytes()
        assert not target.is_relative_to(paths.audio_root)
        with pytest.raises(FmaCatalogError, match="--overwrite"):
            make_query(paths.audio_root, paths.output_root)
        chosen = prepared.manifest["selected_tracks"][-1]
        make_query(paths.audio_root, paths.output_root, chosen["track_id"], overwrite=True)
        assert target.read_bytes() == (paths.audio_root / chosen["audio_path"]).read_bytes()
        with pytest.raises(FmaCatalogError, match="no está en el catálogo"):
            make_query(paths.audio_root, paths.output_root, 3, overwrite=True)
        # La query no se registra en SoupDB.
        catalog = open_catalog(paths.db_path, paths.index_config())
        try:
            songs = CopyrightCatalog(catalog).list_songs()
            assert len(songs) == prepared.manifest["number_of_songs"]
        finally:
            catalog.close()
    finally:
        target.unlink(missing_ok=True)


def test_make_query_requires_prepared_catalog(fma_dataset, tmp_path):
    audio, _ = fma_dataset
    with pytest.raises(FmaCatalogError, match="prepare"):
        make_query(audio, tmp_path / "nothing")


def test_cli_summary_and_errors(prepared, fma_dataset, capsys):
    audio, csv_path = fma_dataset
    argv = [
        "prepare",
        "--audio-root",
        str(audio),
        "--tracks-csv",
        str(csv_path),
        "--output-root",
        str(prepared.paths.output_root),
        "--per-genre",
        "2",
    ]
    assert fma_catalog.main(argv) == 0
    out = capsys.readouterr().out
    assert "FMA catalog reused" in out
    assert "Songs: 16" in out
    assert f"Segments: {prepared.manifest['number_of_segments']}" in out
    assert "Genres: 8" in out
    assert "HNSW: ready" in out
    assert "COPYRIGHT_DATASET_ROOT" in out
    assert len(out.splitlines()) < 40

    assert fma_catalog.main([*argv[:-1], "0"]) == 1
    assert "per_genre" in capsys.readouterr().err


def test_smoke_query_identical_track_ranks_first(prepared):
    """Smoke test: una copia exacta de una pista catalogada sale primera con score ≈ 1."""
    paths = prepared.paths
    chosen = next(t for t in prepared.manifest["selected_tracks"] if t["segment_count"] == 3)
    query = make_query(paths.audio_root, paths.output_root, chosen["track_id"], overwrite=True)
    settings = RuntimeSettings.from_env(paths.env())
    try:
        with TestClient(runtime.build_app(settings)) as client:
            with query.open("rb") as audio:
                response = client.post(
                    "/api/v1/analyze", files={"file": (query.name, audio, "audio/mpeg")}
                )
    finally:
        query.unlink(missing_ok=True)
    assert response.status_code == 200, response.text
    best = response.json()["matches"][0]
    assert best["song_id"] == chosen["song_id"]
    assert best["title"] == chosen["title"]
    assert best["is_best_run"] is True
    assert best["segment_count"] == chosen["segment_count"]
    assert best["alignment_offset_seconds"] == pytest.approx(0.0)
    assert best["score"] > 0.99
