from pathlib import Path

import pytest

from engine.algorithms.external_sort import (
    merge_sorted_runs,
    read_sorted_run,
    write_sorted_run,
)
from engine.common.record import Record


def record(value: int) -> Record:
    return Record(data=str(value).encode())


def compare(left: Record, right: Record) -> int:
    a, b = int(left.data), int(right.data)
    return (a > b) - (a < b)


def test_merge_multiple_passes_preserves_inputs_and_duplicates(tmp_path: Path) -> None:
    paths = [
        write_sorted_run(tmp_path / f"{i}.run", [record(i), record(20)], compare) for i in range(9)
    ]
    original = {p: p.read_bytes() for p in paths}
    work = tmp_path / "work"
    work.mkdir()
    result = list(merge_sorted_runs(paths, compare, fan_in=2, temp_dir=work))
    assert [int(r.data) for r in result] == list(range(9)) + [20] * 9
    assert {p: p.read_bytes() for p in paths} == original
    assert list(work.iterdir()) == []


def test_writer_checks_order_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "run"
    with pytest.raises(ValueError, match="sorted"):
        write_sorted_run(path, [record(2), record(1)], compare)
    assert not path.exists()
    write_sorted_run(path, [record(1)], compare)
    with pytest.raises(FileExistsError):
        write_sorted_run(path, [], compare)
    assert [r.data for r in read_sorted_run(path)] == [b"1"]


@pytest.mark.parametrize("payload", [b"\x01", b"\x02\x00\x00\x00x"])
def test_truncated_run_rejected(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "bad"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="truncated"):
        list(read_sorted_run(path))


def test_early_close_cleans_temp_and_releases_input(tmp_path: Path) -> None:
    paths = [write_sorted_run(tmp_path / str(i), [record(i)], compare) for i in range(5)]
    work = tmp_path / "work"
    work.mkdir()
    merged = merge_sorted_runs(paths, compare, fan_in=2, temp_dir=work)
    assert next(merged).data == b"0"
    merged.close()
    assert list(work.iterdir()) == []
    # On Windows this also detects leaked handles.
    for path in paths:
        path.unlink()


def test_merge_failure_releases_already_open_inputs(tmp_path: Path) -> None:
    first = write_sorted_run(tmp_path / "first", [record(1)], compare)
    with pytest.raises(FileNotFoundError):
        list(merge_sorted_runs([first, tmp_path / "missing"], compare, temp_dir=tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["first"]
    first.unlink()


def test_empty_single_run_and_invalid_fan_in(tmp_path: Path) -> None:
    assert list(merge_sorted_runs([], compare)) == []
    path = write_sorted_run(tmp_path / "empty", [], compare)
    assert list(merge_sorted_runs([path], compare)) == []
    for fan_in in (0, 1, True, 2.5):
        with pytest.raises(ValueError, match="fan_in"):
            list(merge_sorted_runs([], compare, fan_in=fan_in))
