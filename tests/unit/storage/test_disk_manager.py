from pathlib import Path

import pytest

from engine.storage.disk_manager import DiskManager


def test_creates_file_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    DiskManager(path, page_size=4096)
    assert path.exists()


def test_allocate_page_returns_zero_first(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    assert dm.allocate_page() == 0


def test_allocate_page_returns_increasing_unique_ids(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    ids = [dm.allocate_page() for _ in range(3)]
    assert ids == [0, 1, 2]


def test_allocate_page_extends_file_by_page_size(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=4096)
    dm.allocate_page()
    assert path.stat().st_size == 4096
    dm.allocate_page()
    assert path.stat().st_size == 8192


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    data = bytes(range(256)) * 16  # 4096 bytes
    dm.write_page(page_id, data)
    assert dm.read_page(page_id) == data


def test_write_page_rejects_undersized_data(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    with pytest.raises(ValueError):
        dm.write_page(page_id, b"\x00" * 100)


def test_write_page_rejects_oversized_data(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    with pytest.raises(ValueError):
        dm.write_page(page_id, b"\x00" * 5000)


def test_counters_start_at_zero(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    assert dm.reads == 0
    assert dm.writes == 0


def test_allocate_page_increments_writes(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.allocate_page()
    assert dm.writes == 1
    dm.allocate_page()
    assert dm.writes == 2


def test_write_page_increments_writes(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    writes_before = dm.writes
    dm.write_page(page_id, b"\x00" * 4096)
    assert dm.writes == writes_before + 1


def test_read_page_increments_reads(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    assert dm.reads == 0
    dm.read_page(page_id)
    assert dm.reads == 1
    dm.read_page(page_id)
    assert dm.reads == 2


def test_read_page_does_not_increment_writes(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    writes_before = dm.writes
    dm.read_page(page_id)
    assert dm.writes == writes_before


def test_write_page_does_not_increment_reads(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    reads_before = dm.reads
    dm.write_page(page_id, b"\x00" * 4096)
    assert dm.reads == reads_before


def test_data_survives_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=4096)
    page_id = dm.allocate_page()
    data = bytes(range(256)) * 16  # 4096 bytes
    dm.write_page(page_id, data)
    dm.close()

    reopened = DiskManager(path, page_size=4096)
    assert reopened.read_page(page_id) == data


def test_allocate_page_after_reopen_continues_sequence(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=4096)
    dm.allocate_page()
    dm.allocate_page()
    dm.close()

    reopened = DiskManager(path, page_size=4096)
    assert reopened.allocate_page() == 2


def test_close_is_idempotent(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.close()
    dm.close()


def test_read_page_after_close_raises_value_error(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    dm.close()
    with pytest.raises(ValueError):
        dm.read_page(page_id)


def test_write_page_after_close_raises_value_error(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    dm.close()
    with pytest.raises(ValueError):
        dm.write_page(page_id, b"\x00" * 4096)


def test_allocate_page_after_close_raises_value_error(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.close()
    with pytest.raises(ValueError):
        dm.allocate_page()


def test_rejects_zero_page_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DiskManager(tmp_path / "data.db", page_size=0)


def test_rejects_negative_page_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DiskManager(tmp_path / "data.db", page_size=-1)


def test_rejects_existing_file_with_size_not_multiple_of_page_size(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    path.write_bytes(b"\x00" * 100)  # not a multiple of 4096
    with pytest.raises(ValueError):
        DiskManager(path, page_size=4096)


def test_read_page_rejects_negative_page_id(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.allocate_page()
    with pytest.raises(ValueError):
        dm.read_page(-1)


def test_write_page_rejects_negative_page_id(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.allocate_page()
    with pytest.raises(ValueError):
        dm.write_page(-1, b"\x00" * 4096)


def test_read_page_rejects_unallocated_page_id(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.allocate_page()
    with pytest.raises(ValueError):
        dm.read_page(1)


def test_write_page_rejects_unallocated_page_id(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.allocate_page()
    with pytest.raises(ValueError):
        dm.write_page(1, b"\x00" * 4096)


def test_read_page_raises_on_short_read(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    page_id = dm.allocate_page()
    dm._file.truncate(100)  # simulate a corrupted/truncated page on disk
    with pytest.raises(ValueError):
        dm.read_page(page_id)


def test_page_size_property(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    assert dm.page_size == 4096


def test_page_count_starts_at_zero(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    assert dm.page_count == 0


def test_page_count_increments_with_allocate_page(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=4096)
    dm.allocate_page()
    assert dm.page_count == 1
    dm.allocate_page()
    assert dm.page_count == 2


def test_page_count_reflects_existing_file_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=4096)
    dm.allocate_page()
    dm.allocate_page()
    dm.close()

    reopened = DiskManager(path, page_size=4096)
    assert reopened.page_count == 2
