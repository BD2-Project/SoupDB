"""Conformance suite for every RankedIndex implementation.

Each case knows how to build an index over a fixed dataset, how to reopen it
from the same path and which options make it answer exactly (for approximate
indexes, probing everything), so the same tests apply to all of them.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.vector.brute_force import BruteForceIndex
from tests.fakes.fake_ranked_index import FakeRankedIndex, text_scorer, vector_scorer

N_ITEMS = 60
DIM = 8


@dataclass
class Case:
    build: Callable[[Path, list[tuple[Any, RID]]], RankedIndex]
    reopen: Callable[[Path], RankedIndex]
    items: list[tuple[Any, RID]]
    query: Any
    opts: dict[str, Any] = field(default_factory=dict)


def _vectors() -> list[tuple[Any, RID]]:
    rng = random.Random(7)
    return [
        ([rng.uniform(-1, 1) for _ in range(DIM)], RID(i // 10, i % 10)) for i in range(N_ITEMS)
    ]


def _texts() -> list[tuple[Any, RID]]:
    rng = random.Random(11)
    words = ["index", "vector", "page", "tree", "hash", "query", "disk", "buffer"]
    return [(" ".join(rng.choices(words, k=6)), RID(i, 0)) for i in range(N_ITEMS)]


def _fake_build(scorer):
    def build(path: Path, items: list[tuple[Any, RID]]) -> RankedIndex:
        index = FakeRankedIndex(scorer, path / "fake.json")
        for item, rid in items:
            index.add(item, rid)
        return index

    return build


def _brute_build(path: Path, items: list[tuple[Any, RID]]) -> RankedIndex:
    index = BruteForceIndex.open(path / "bf.db", dim=DIM, page_size=512, buffer_capacity=2)
    for vector, rid in items:
        index.insert(vector, rid)
    return index


CASES: dict[str, Callable[[], Case]] = {
    "fake_vector": lambda: Case(
        build=_fake_build(vector_scorer),
        reopen=lambda path: FakeRankedIndex(vector_scorer, path / "fake.json"),
        items=_vectors(),
        query=[0.1] * DIM,
        opts={"metric": "cosine"},
    ),
    "fake_text": lambda: Case(
        build=_fake_build(text_scorer),
        reopen=lambda path: FakeRankedIndex(text_scorer, path / "fake.json"),
        items=_texts(),
        query="vector index",
        opts={"method": "bm25"},
    ),
    "brute_force": lambda: Case(
        build=_brute_build,
        reopen=lambda path: BruteForceIndex.open(path / "bf.db", page_size=512),
        items=_vectors(),
        query=[0.1] * DIM,
        opts={"metric": "cosine"},
    ),
}


@pytest.fixture(params=list(CASES))
def case(request) -> Case:
    return CASES[request.param]()


@pytest.fixture
def index(case: Case, tmp_path: Path):
    idx = case.build(tmp_path, case.items)
    yield idx
    idx.close()


def test_results_are_sorted_by_descending_score(index, case: Case) -> None:
    result = index.top_k(case.query, 10, **case.opts)

    scores = [score for _rid, score in result]
    assert len(result) == 10
    assert scores == sorted(scores, reverse=True)


def test_results_are_inserted_rids(index, case: Case) -> None:
    inserted = {rid for _item, rid in case.items}

    result = index.top_k(case.query, 10, **case.opts)

    assert {rid for rid, _score in result} <= inserted


def test_k_larger_than_total_returns_everything(index, case: Case) -> None:
    result = index.top_k(case.query, N_ITEMS * 3, **case.opts)

    assert sorted(rid for rid, _score in result) == sorted(rid for _item, rid in case.items)


def test_k_zero_returns_empty(index, case: Case) -> None:
    assert index.top_k(case.query, 0, **case.opts) == []


def test_negative_k_is_rejected(index, case: Case) -> None:
    with pytest.raises(ValueError):
        index.top_k(case.query, -1, **case.opts)


def test_unknown_option_is_rejected(index, case: Case) -> None:
    with pytest.raises((ValueError, TypeError)):
        index.top_k(case.query, 5, bogus_option=1)


def test_persistence_after_close_and_reopen(case: Case, tmp_path: Path) -> None:
    idx = case.build(tmp_path, case.items)
    before = idx.top_k(case.query, 10, **case.opts)
    idx.close()

    reopened = case.reopen(tmp_path)
    try:
        after = reopened.top_k(case.query, 10, **case.opts)
    finally:
        reopened.close()

    assert [rid for rid, _ in after] == [rid for rid, _ in before]
    assert [score for _, score in after] == pytest.approx([score for _, score in before])
