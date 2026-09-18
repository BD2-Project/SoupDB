from pathlib import Path

import pytest

from engine.common.rid import RID
from engine.indexes.bplus_tree import BPlusTree
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 128


def _make_tree(
    tmp_path: Path,
    *,
    buffer_capacity: int = 2,
) -> tuple[BPlusTree, DiskManager, BufferManager]:
    dm = DiskManager(tmp_path / "index.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=buffer_capacity)
    return BPlusTree(dm, bm), dm, bm


def test_new_tree_creates_metadata_and_root_leaf(tmp_path: Path) -> None:
    tree, dm, _bm = _make_tree(tmp_path)

    assert dm.page_count == 2
    assert tree.search(10) == []


def test_insert_and_search(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)
    rid = RID(4, 2)

    tree.insert(10, rid)

    assert tree.search(10) == [rid]


def test_search_missing_key_returns_empty(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    tree.insert(10, RID(1, 1))

    assert tree.search(99) == []


def test_duplicate_keys_return_all_rids(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)
    rids = [RID(0, index) for index in range(3)]

    for rid in rids:
        tree.insert(10, rid)

    assert tree.search(10) == rids


def test_entries_are_returned_in_key_order(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    tree.insert(30, RID(0, 3))
    tree.insert(10, RID(0, 1))
    tree.insert(20, RID(0, 2))

    assert tree.range_search(0, 100) == [
        RID(0, 1),
        RID(0, 2),
        RID(0, 3),
    ]


def test_range_search_is_inclusive(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    for key in (10, 20, 30, 40):
        tree.insert(key, RID(0, key))

    assert tree.range_search(20, 30) == [
        RID(0, 20),
        RID(0, 30),
    ]


def test_remove_specific_rid_preserves_duplicate(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)
    first = RID(1, 1)
    second = RID(1, 2)

    tree.insert(5, first)
    tree.insert(5, second)

    assert tree.remove(5, first) == 1
    assert tree.search(5) == [second]


def test_remove_all_under_key(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    tree.insert(7, RID(0, 1))
    tree.insert(7, RID(0, 2))
    tree.insert(8, RID(0, 3))

    assert tree.remove(7) == 2
    assert tree.search(7) == []
    assert tree.search(8) == [RID(0, 3)]


def test_remove_missing_key_returns_zero(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    assert tree.remove(100) == 0


def test_leaf_split_preserves_all_entries(tmp_path: Path) -> None:
    tree, dm, _bm = _make_tree(tmp_path)

    expected = {}
    for key in range(18):
        rid = RID(key // 4, key)
        expected[key] = rid
        tree.insert(key, rid)

    assert dm.page_count > 2

    for key, rid in expected.items():
        assert tree.search(key) == [rid]


def test_range_search_crosses_leaf_boundaries(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    for key in range(18):
        tree.insert(key, RID(0, key))

    assert tree.range_search(4, 13) == [RID(0, key) for key in range(4, 14)]


def test_duplicate_key_can_span_multiple_leaves(tmp_path: Path) -> None:
    tree, dm, _bm = _make_tree(tmp_path)
    rids = [RID(index // 3, index) for index in range(14)]

    for rid in rids:
        tree.insert(7, rid)

    assert dm.page_count > 3
    assert tree.search(7) == rids


def test_remove_all_duplicates_across_multiple_leaves(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)
    rids = [RID(index // 3, index) for index in range(14)]

    for rid in rids:
        tree.insert(7, rid)

    assert tree.remove(7) == len(rids)
    assert tree.search(7) == []


def test_split_structure_persists_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "index.db"

    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    tree = BPlusTree(dm, bm)

    for key in range(18):
        tree.insert(key, RID(0, key))

    tree.close()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = BPlusTree(dm2, bm2)

    assert reopened.range_search(0, 17) == [RID(0, key) for key in range(18)]


def test_persistence_after_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "index.db"

    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    tree = BPlusTree(dm, bm)

    tree.insert(10, RID(1, 2))
    tree.insert(20, RID(3, 4))
    tree.close()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = BPlusTree(dm2, bm2)

    assert reopened.search(10) == [RID(1, 2)]
    assert reopened.search(20) == [RID(3, 4)]


def test_close_flushes_dirty_pages(tmp_path: Path) -> None:
    path = tmp_path / "index.db"

    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    tree = BPlusTree(dm, bm)

    tree.insert(10, RID(1, 1))
    tree.close()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    reopened = BPlusTree(dm2, bm2)

    assert reopened.search(10) == [RID(1, 1)]


def test_operations_after_close_are_rejected(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)
    tree.close()

    with pytest.raises(RuntimeError):
        tree.search(10)


def test_single_entry_too_large_for_leaf_is_rejected(tmp_path: Path) -> None:
    tree, _dm, _bm = _make_tree(tmp_path)

    with pytest.raises(ValueError):
        tree.insert("x" * PAGE_SIZE, RID(0, 0))
