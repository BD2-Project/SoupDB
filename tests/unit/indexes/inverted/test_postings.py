from pathlib import Path

import pytest

from engine.indexes.inverted.postings import PostingsFile


def test_binary_postings_roundtrip_and_multiple_lists(tmp_path: Path) -> None:
    path = tmp_path / "postings.bin"
    file = PostingsFile.create(path)
    a = file.append([(0, 2), (9, 7)])
    b = file.append([])
    file.close()
    file.close()
    reopened = PostingsFile.load(path)
    assert reopened.read(*a, df=2) == [(0, 2), (9, 7)]
    assert reopened.read(*b, df=0) == []
    with pytest.raises(ValueError):
        reopened.append([(1, 1)])
    reopened.close()


@pytest.mark.parametrize(
    "postings", [[(1, 1), (1, 2)], [(2, 1), (0, 1)], [(-1, 1)], [(0, 0)], [(0, True)], [(2**64, 1)]]
)
def test_invalid_postings_are_rejected_without_writing(tmp_path, postings) -> None:
    file = PostingsFile.create(tmp_path / "postings.bin")
    with pytest.raises(ValueError):
        file.append(postings)
    assert file.append([(0, 1)])[0] == 8
    file.close()


def test_reject_missing_header_version_and_out_of_bounds(tmp_path: Path) -> None:
    path = tmp_path / "postings.bin"
    with pytest.raises(FileNotFoundError):
        PostingsFile.load(path)
    assert not path.exists()
    path.write_bytes(b"SPST\x02\x00\x00\x00")
    with pytest.raises(ValueError):
        PostingsFile.load(path)
    path.unlink()
    file = PostingsFile.create(path)
    offset, length = file.append([(0, 1)])
    for bad_offset, bad_length, df in [
        (0, length, 1),
        (offset, length, 2),
        (offset + 100, length, 1),
    ]:
        with pytest.raises(ValueError):
            file.read(bad_offset, bad_length, df=df)
    file.close()


def test_corrupt_count_and_truncated_payload(tmp_path: Path) -> None:
    path = tmp_path / "postings.bin"
    file = PostingsFile.create(path)
    location = file.append([(0, 2)])
    file.close()
    data = bytearray(path.read_bytes())
    data[8] = 2
    path.write_bytes(data)
    reopened = PostingsFile.load(path)
    with pytest.raises(ValueError):
        reopened.read(*location, df=1)
    reopened.close()
    path.write_bytes(data[:-1])
    reopened = PostingsFile.load(path)
    with pytest.raises(ValueError):
        reopened.read(*location, df=1)
    reopened.close()
