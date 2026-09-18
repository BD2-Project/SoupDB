from pathlib import Path

import pytest

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes import _hash_page as codec
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager


def open_index(tmp_path: Path, **kwargs) -> ExtendibleHash:
    return ExtendibleHash.open(tmp_path / "idx.db", **kwargs)


def test_insert_and_search(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    rid = RID(page_id=3, slot=7)
    index.insert(42, rid)
    assert index.search(42) == [rid]
    index.close()


def test_search_missing_key_returns_empty(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    assert index.search(999) == []
    index.close()


def test_duplicate_keys_return_all_rids(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    rids = [RID(page_id=1, slot=i) for i in range(3)]
    for rid in rids:
        index.insert(7, rid)
    assert sorted(index.search(7)) == sorted(rids)
    index.close()


def test_remove_single_rid(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    first, second = RID(1, 1), RID(1, 2)
    index.insert(5, first)
    index.insert(5, second)
    assert index.remove(5, first) == 1
    assert index.search(5) == [second]
    index.close()


def test_remove_all_rids_under_key(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    for slot in range(4):
        index.insert(5, RID(0, slot))
    assert index.remove(5) == 4
    assert index.search(5) == []
    index.close()


def test_remove_missing_key_returns_zero(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    assert index.remove(123) == 0
    index.close()


def test_does_not_support_range_search(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    assert index.supports_range is False
    with pytest.raises(UnsupportedOperation):
        index.range_search(1, 10)
    index.close()


def test_string_keys(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    rid = RID(2, 2)
    index.insert("extendible hashing", rid)
    assert index.search("extendible hashing") == [rid]
    assert index.search("otro término") == []
    index.close()


def test_int_and_string_keys_do_not_collide(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    index.insert(1, RID(0, 1))
    index.insert("1", RID(0, 2))
    assert index.search(1) == [RID(0, 1)]
    assert index.search("1") == [RID(0, 2)]
    index.close()


def test_rejects_unsupported_key_type(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    with pytest.raises(ValueError):
        index.insert(3.5, RID(0, 0))
    index.close()


def test_many_keys_split_buckets_and_grow_directory(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    for key in range(500):
        index.insert(key, RID(page_id=key, slot=0))

    assert index.global_depth > 0
    for key in range(500):
        assert index.search(key) == [RID(page_id=key, slot=0)]
    index.close()


def test_split_uses_suffix_labels(tmp_path: Path) -> None:
    """Tras dividir, la entrada con el bit nuevo en 1 apunta a otro bucket que su gemela."""
    index = open_index(tmp_path)
    for key in range(400):
        index.insert(key, RID(page_id=key, slot=0))

    depth = index.global_depth
    assert depth >= 1
    half = 1 << (depth - 1)
    buckets = [index._directory_slot(slot) for slot in range(1 << depth)]
    assert any(buckets[slot] != buckets[slot + half] for slot in range(half))
    index.close()


def test_data_survives_close_and_reopen(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    for key in range(300):
        index.insert(key, RID(page_id=key, slot=1))
    index.insert("papers", RID(page_id=9, slot=9))
    depth = index.global_depth
    index.close()

    reopened = open_index(tmp_path)
    assert reopened.global_depth == depth
    for key in range(300):
        assert reopened.search(key) == [RID(page_id=key, slot=1)]
    assert reopened.search("papers") == [RID(page_id=9, slot=9)]
    reopened.close()


def test_overflow_chain_when_depth_is_capped(tmp_path: Path) -> None:
    """Sin margen para duplicar el directorio, las claves siguen recuperables vía overflow."""
    index = open_index(tmp_path, max_global_depth=0)
    rids = [RID(page_id=0, slot=slot) for slot in range(400)]
    for rid in rids:
        index.insert(1, rid)

    assert sorted(index.search(1)) == sorted(rids)
    assert index.remove(1) == len(rids)
    index.close()


def test_rejects_key_larger_than_a_page(tmp_path: Path) -> None:
    index = open_index(tmp_path, page_size=4096)
    with pytest.raises(ValueError):
        index.insert("x" * 5000, RID(0, 0))
    index.close()


def test_accepts_injected_managers(tmp_path: Path) -> None:
    disk_manager = DiskManager(tmp_path / "idx.db", page_size=4096)
    buffer_manager = BufferManager(disk_manager, capacity=8)
    index = ExtendibleHash(disk_manager, buffer_manager)
    index.insert(1, RID(0, 0))
    assert index.search(1) == [RID(0, 0)]
    index.close()
    disk_manager.close()


def test_all_io_goes_through_the_disk_manager(tmp_path: Path) -> None:
    """Los contadores del DiskManager son la métrica que evalúa el curso."""
    disk_manager = DiskManager(tmp_path / "idx.db", page_size=4096)
    buffer_manager = BufferManager(disk_manager, capacity=4)
    index = ExtendibleHash(disk_manager, buffer_manager)
    for key in range(200):
        index.insert(key, RID(page_id=key, slot=0))
    index.close()

    assert disk_manager.writes > 0
    assert disk_manager.reads > 0
    disk_manager.close()


def test_close_is_idempotent(tmp_path: Path) -> None:
    index = open_index(tmp_path)
    index.close()
    index.close()


def test_page_codec_reports_bucket_depth_and_overflow() -> None:
    page = codec.new_bucket(4096, local_depth=2)
    assert codec.local_depth(page) == 2
    assert codec.overflow_page_id(page) == codec.NO_OVERFLOW
    assert codec.entry_count(page) == 0

    assert codec.add_entry(page, b"\x00key", RID(1, 2)) is True
    assert codec.entry_count(page) == 1
    assert list(codec.iter_entries(page)) == [(b"\x00key", RID(1, 2))]

    codec.set_overflow_page_id(page, 5)
    assert codec.overflow_page_id(page) == 5
    assert codec.local_depth(page) == 2
