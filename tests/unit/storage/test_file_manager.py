"""Tests for the file manager wrapper."""

from pathlib import Path

from engine.storage.file_manager import FileManager


def _file_exists(root: Path, name: str) -> bool:
    return (root / f"{name}.db").exists()


def test_storage_creates_file_on_first_touch(tmp_path: Path) -> None:
    fm = FileManager(tmp_path, page_size=64)
    dm, bm = fm.storage("t1")
    assert dm.page_count == 0
    assert _file_exists(tmp_path, "t1")
    fm.close()


def test_storage_reuses_same_managers(tmp_path: Path) -> None:
    fm = FileManager(tmp_path, page_size=64)
    dm_a, bm_a = fm.storage("t1")
    dm_b, bm_b = fm.storage("t1")
    assert dm_a is dm_b
    assert bm_a is bm_b
    fm.close()


def test_two_files_are_independent(tmp_path: Path) -> None:
    fm = FileManager(tmp_path, page_size=64)
    dm_a, bm_a = fm.storage("a")
    dm_b, bm_b = fm.storage("b")
    dm_a.allocate_page()
    assert dm_a.page_count == 1
    assert dm_b.page_count == 0
    fm.close()


def test_pages_persist_across_reopen(tmp_path: Path) -> None:
    fm = FileManager(tmp_path, page_size=64)
    dm, bm = fm.storage("a")
    dm.allocate_page()
    dm.write_page(0, bytes(64))
    fm.close()

    reopened = FileManager(tmp_path, page_size=64)
    dm2, _bm2 = reopened.storage("a")
    assert dm2.page_count == 1
    assert dm2.read_page(0) == bytes(64)
    reopened.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    fm = FileManager(tmp_path, page_size=64)
    fm.storage("a")
    fm.close()
    fm.close()
