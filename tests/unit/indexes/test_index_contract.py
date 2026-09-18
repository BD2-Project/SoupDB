"""Conformance suite for every Index implementation.

The suite parametrizes over all implementations; real disk-backed indexes
(BPlusTree, ExtendibleHash) join the params when they exist.
"""

from collections import defaultdict

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes.extendible_hash import ExtendibleHash
from tests.fakes.fake_index import FakeIndex


def _build(kind, tmp_path):
    if kind is ExtendibleHash:
        return ExtendibleHash.open(tmp_path / "idx.db")
    return kind()


@pytest.fixture(params=[FakeIndex, ExtendibleHash])
def index(request, tmp_path):
    idx = _build(request.param, tmp_path)
    yield idx
    idx.close()


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


def test_remove_and_verify_gone(index) -> None:
    rid = RID(1, 1)
    index.insert(5, rid)
    assert index.remove(5, rid) == 1
    assert index.search(5) == []


def test_remove_all_under_key(index) -> None:
    for i in range(2):
        index.insert(7, RID(0, i))
    assert index.remove(7) == 2
    assert index.search(7) == []


def test_persistence(tmp_path) -> None:
    """Cerrar, reabrir desde el mismo path y verificar que los datos siguen ahí.

    Solo aplica a los índices en disco: FakeIndex vive en memoria.
    """
    idx = ExtendibleHash.open(tmp_path / "persist.db")
    rid = RID(page_id=4, slot=2)
    idx.insert("chunks", rid)
    idx.insert(11, RID(page_id=1, slot=1))
    idx.close()

    reopened = ExtendibleHash.open(tmp_path / "persist.db")
    assert reopened.search("chunks") == [rid]
    assert reopened.search(11) == [RID(page_id=1, slot=1)]
    reopened.close()


def test_range_search_is_rejected_when_unsupported(index) -> None:
    if index.supports_range:
        pytest.skip("el índice soporta rangos")
    with pytest.raises(UnsupportedOperation):
        index.range_search(0, 10)


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
