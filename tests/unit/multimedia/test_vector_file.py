import struct

import pytest

from engine.multimedia.vector_file import (
    MAGIC,
    VERSION,
    VectorRecord,
    iter_vectors,
    random_vectors,
    write_vectors,
)


def _as_float32(values):
    return struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values))


def test_round_trip_preserves_ids_paths_and_float32_values(tmp_path):
    file = tmp_path / "vectores.bin"
    records = [
        VectorRecord(7, "img/gato.jpg", [0.1, -2.5, 3.0]),
        VectorRecord(-1, "audio/canción ñ.wav", [1e-3, 0.0, -7.25]),
        VectorRecord(2**40, "", [1.0, 2.0, 3.0]),
    ]
    write_vectors(file, records, dimension=3, count=len(records))

    read = list(iter_vectors(file))

    assert [(r.id, r.path) for r in read] == [(r.id, r.path) for r in records]
    assert [r.vector for r in read] == [_as_float32(r.vector) for r in records]


def test_header_stores_count_and_dimension(tmp_path):
    file = tmp_path / "vectores.bin"
    write_vectors(file, random_vectors(5, 4, seed=1), dimension=4, count=5)

    header = struct.unpack("<4sIII", file.read_bytes()[:16])

    assert header == (MAGIC, VERSION, 5, 4)


def test_random_vectors_is_deterministic_for_a_seed():
    assert list(random_vectors(10, 8, seed=42)) == list(random_vectors(10, 8, seed=42))
    assert list(random_vectors(10, 8, seed=42)) != list(random_vectors(10, 8, seed=43))


def test_streams_100k_records(tmp_path):
    file = tmp_path / "vectores.bin"
    n = 100_000
    write_vectors(file, random_vectors(n, 4, seed=0), dimension=4, count=n)

    records = iter_vectors(file)
    assert iter(records) is records

    seen = 0
    for record in records:
        assert record.id == seen
        assert len(record.vector) == 4
        seen += 1
    assert seen == n


def test_truncated_file_raises_eof(tmp_path):
    file = tmp_path / "vectores.bin"
    write_vectors(file, random_vectors(3, 4, seed=0), dimension=4, count=3)
    file.write_bytes(file.read_bytes()[:-1])

    with pytest.raises(EOFError):
        list(iter_vectors(file))


@pytest.mark.parametrize(
    "header",
    [
        struct.pack("<4sIII", b"XXXX", VERSION, 0, 4),
        struct.pack("<4sIII", MAGIC, 99, 0, 4),
        struct.pack("<4sIII", MAGIC, VERSION, 0, 0),
    ],
    ids=["bad-magic", "bad-version", "zero-dimension"],
)
def test_invalid_header_raises(tmp_path, header):
    file = tmp_path / "vectores.bin"
    file.write_bytes(header)

    with pytest.raises(ValueError):
        list(iter_vectors(file))


def test_writer_rejects_vector_of_wrong_dimension(tmp_path):
    file = tmp_path / "vectores.bin"
    records = [VectorRecord(0, "a", [1.0, 2.0])]

    with pytest.raises(ValueError):
        write_vectors(file, records, dimension=3, count=1)
    assert not file.exists()


@pytest.mark.parametrize("given", [2, 4])
def test_writer_rejects_count_mismatch(tmp_path, given):
    file = tmp_path / "vectores.bin"

    with pytest.raises(ValueError):
        write_vectors(file, random_vectors(given, 2, seed=0), dimension=2, count=3)
    assert not file.exists()
