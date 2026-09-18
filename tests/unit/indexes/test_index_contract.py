"""Conformance suite for every Index implementation."""

from collections import defaultdict

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes.bplus_tree import BPlusTree
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from tests.fakes.fake_index import FakeIndex

PAGE_SIZE = 256


@pytest.fixture(params=["fake", "bplus", "hash"])
def index(request, tmp_path):
    disk_manager = None

    if request.param == "fake":
        idx = FakeIndex()
    else:
        disk_manager = DiskManager(
            tmp_path / "index.db",
            page_size=PAGE_SIZE,
        )
        buffer_manager = BufferManager(
            disk_manager,
            capacity=2,
        )

        if request.param == "bplus":
            idx = BPlusTree(
                disk_manager,
                buffer_manager,
            )
        else:
            idx = ExtendibleHash(
                disk_manager,
                buffer_manager,
            )

    try:
        yield idx
    finally:
        idx.close()

        if disk_manager is not None:
            disk_manager.close()


def test_insert_and_search(index) -> None:
    rid = RID(page_id=0, slot=1)

    index.insert(10, rid)

    assert index.search(10) == [rid]


def test_duplicate_keys_return_all_rids(index) -> None:
    rids = [RID(0, i) for i in range(3)]

    for rid in rids:
        index.insert("a", rid)

    assert sorted(index.search("a")) == sorted(rids)


def test_search_missing_key_returns_empty(index) -> None:
    assert index.search(999) == []


def test_range_search_is_inclusive(index) -> None:
    entries = [
        (10, RID(0, 1)),
        (20, RID(0, 2)),
        (30, RID(0, 3)),
        (40, RID(0, 4)),
    ]

    for key, rid in entries:
        index.insert(key, rid)

    if index.supports_range:
        result = index.range_search(20, 30)
        assert sorted(result) == sorted([RID(0, 2), RID(0, 3)])
    else:
        with pytest.raises(UnsupportedOperation):
            index.range_search(20, 30)


def test_remove_and_verify_gone(index) -> None:
    rid = RID(1, 1)

    index.insert(5, rid)

    assert index.remove(5, rid) == 1
    assert index.search(5) == []


def test_remove_one_rid_preserves_other_duplicates(index) -> None:
    first = RID(1, 1)
    second = RID(1, 2)

    index.insert(5, first)
    index.insert(5, second)

    assert index.remove(5, first) == 1
    assert index.search(5) == [second]


def test_remove_all_under_key(index) -> None:
    for i in range(2):
        index.insert(7, RID(0, i))

    assert index.remove(7) == 2
    assert index.search(7) == []


def test_remove_missing_key_returns_zero(index) -> None:
    assert index.remove(999) == 0


def test_remove_missing_rid_returns_zero_without_modifying_key(index) -> None:
    existing = RID(0, 1)
    missing = RID(0, 2)

    index.insert(10, existing)

    assert index.remove(10, missing) == 0
    assert index.search(10) == [existing]


def test_index_without_range_support_raises_unsupported_operation() -> None:
    index = FakeIndex(supports_range=False)

    try:
        index.insert(10, RID(0, 1))

        assert index.supports_range is False

        with pytest.raises(UnsupportedOperation):
            index.range_search(0, 20)
    finally:
        index.close()


@given(st.lists(st.integers(min_value=0, max_value=1000), max_size=200))
def test_fuzz_against_oracle(keys: list[int]) -> None:
    idx = FakeIndex()
    oracle: dict[int, list[RID]] = defaultdict(list)

    for key in keys:
        rid = RID(0, len(oracle[key]))
        idx.insert(key, rid)
        oracle[key].append(rid)

    sample = set(keys) | {2000}

    for key in sample:
        assert sorted(idx.search(key)) == sorted(oracle[key])

    idx.close()
