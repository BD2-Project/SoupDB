from pathlib import Path

import pytest

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 256


def _make_index(
    tmp_path: Path,
    *,
    buffer_capacity: int = 2,
) -> tuple[ExtendibleHash, DiskManager, BufferManager]:
    dm = DiskManager(tmp_path / "hash.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=buffer_capacity)
    return ExtendibleHash(dm, bm), dm, bm


def test_new_index_creates_metadata_directory_and_bucket(tmp_path: Path) -> None:
    index, dm, _bm = _make_index(tmp_path)

    assert dm.page_count == 3
    assert index.search(10) == []


def test_insert_and_search(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)
    rid = RID(1, 2)

    index.insert(10, rid)

    assert index.search(10) == [rid]


def test_search_missing_key_returns_empty(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)

    index.insert(10, RID(1, 1))

    assert index.search(999) == []


def test_duplicate_keys_return_all_rids(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)
    rids = [RID(0, position) for position in range(3)]

    for rid in rids:
        index.insert("a", rid)

    assert index.search("a") == rids


def test_range_search_is_not_supported(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)

    assert index.supports_range is False

    with pytest.raises(UnsupportedOperation):
        index.range_search(1, 10)


def test_remove_specific_rid_preserves_duplicate(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)
    first = RID(1, 1)
    second = RID(1, 2)

    index.insert(5, first)
    index.insert(5, second)

    assert index.remove(5, first) == 1
    assert index.search(5) == [second]


def test_remove_all_under_key(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)

    index.insert(7, RID(0, 1))
    index.insert(7, RID(0, 2))
    index.insert(8, RID(0, 3))

    assert index.remove(7) == 2
    assert index.search(7) == []
    assert index.search(8) == [RID(0, 3)]


def test_remove_missing_key_returns_zero(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)

    assert index.remove(100) == 0


def test_remove_missing_rid_does_not_modify_bucket(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)
    existing = RID(0, 1)

    index.insert(10, existing)

    assert index.remove(10, RID(0, 99)) == 0
    assert index.search(10) == [existing]


def test_persistence_after_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "hash.db"

    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    index = ExtendibleHash(dm, bm)

    index.insert(10, RID(1, 2))
    index.insert("hello", RID(3, 4))
    index.close()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = ExtendibleHash(dm2, bm2)

    assert reopened.search(10) == [RID(1, 2)]
    assert reopened.search("hello") == [RID(3, 4)]


def test_close_flushes_dirty_pages(tmp_path: Path) -> None:
    path = tmp_path / "hash.db"

    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    index = ExtendibleHash(dm, bm)

    index.insert(10, RID(1, 1))
    index.close()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = ExtendibleHash(dm2, bm2)

    assert reopened.search(10) == [RID(1, 1)]


def test_operations_after_close_are_rejected(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)
    index.close()

    with pytest.raises(RuntimeError):
        index.search(10)


def test_works_with_single_buffer_frame(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(
        tmp_path,
        buffer_capacity=1,
    )

    index.insert(10, RID(0, 1))
    index.insert(20, RID(0, 2))

    assert index.search(10) == [RID(0, 1)]
    assert index.search(20) == [RID(0, 2)]


def test_bucket_split_preserves_all_entries(tmp_path: Path) -> None:
    index, dm, _bm = _make_index(tmp_path)

    expected = {}

    for key in range(40):
        rid = RID(key // 10, key)
        expected[key] = rid
        index.insert(key, rid)

    assert dm.page_count > 3

    for key, rid in expected.items():
        assert index.search(key) == [rid]


def test_multiple_bucket_splits_preserve_entries(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)

    for key in range(100):
        index.insert(key, RID(0, key))

    for key in range(100):
        assert index.search(key) == [RID(0, key)]


def test_split_structure_persists_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "hash.db"

    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    index = ExtendibleHash(dm, bm)

    for key in range(80):
        index.insert(key, RID(key // 10, key))

    index.close()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = ExtendibleHash(dm2, bm2)

    for key in range(80):
        assert reopened.search(key) == [RID(key // 10, key)]


def test_duplicate_keys_use_overflow_at_maximum_depth(tmp_path: Path) -> None:
    path = tmp_path / "hash.db"

    dm = DiskManager(path, page_size=128)
    bm = BufferManager(dm, capacity=2)
    index = ExtendibleHash(
        dm,
        bm,
        max_global_depth=2,
    )

    expected = [RID(position // 10, position) for position in range(30)]

    for rid in expected:
        index.insert(7, rid)

    assert index.search(7) == expected


def test_overflow_chain_persists_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "hash.db"

    dm = DiskManager(path, page_size=128)
    bm = BufferManager(dm, capacity=2)
    index = ExtendibleHash(
        dm,
        bm,
        max_global_depth=2,
    )

    expected = [RID(0, position) for position in range(30)]

    for rid in expected:
        index.insert(7, rid)

    index.close()
    dm.close()

    dm2 = DiskManager(path, page_size=128)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = ExtendibleHash(dm2, bm2)

    assert reopened.search(7) == expected


def test_remove_all_duplicates_from_overflow_chain(tmp_path: Path) -> None:
    path = tmp_path / "hash.db"

    dm = DiskManager(path, page_size=128)
    bm = BufferManager(dm, capacity=2)
    index = ExtendibleHash(
        dm,
        bm,
        max_global_depth=2,
    )

    expected = [RID(0, position) for position in range(30)]

    for rid in expected:
        index.insert(7, rid)

    assert index.remove(7) == len(expected)
    assert index.search(7) == []


def test_large_entry_is_rejected_without_repeated_splits(tmp_path: Path) -> None:
    index, _dm, _bm = _make_index(tmp_path)

    with pytest.raises(ValueError, match="does not fit"):
        index.insert("x" * PAGE_SIZE, RID(0, 0))
