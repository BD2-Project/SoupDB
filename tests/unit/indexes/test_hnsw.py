"""HNSW index: graph invariants, persistence and recall."""

import random

import pytest

from engine.common.rid import RID
from engine.indexes.vector.hnsw import HNSWIndex
from engine.indexes.vector.metrics import COSINE, EUCLIDEAN
from tests.fakes.fake_ranked_index import FakeRankedIndex, vector_scorer
from tests.unit.indexes.vector_fixtures import clustered_vectors, recall_at_k

DIM = 16


def _build(path, items, **kwargs) -> HNSWIndex:
    index = HNSWIndex(path, dim=DIM, **kwargs)
    for vector, rid in items:
        index.insert(vector, rid)
    return index


def test_empty_index_returns_nothing(tmp_path) -> None:
    index = HNSWIndex(tmp_path / "hnsw", dim=3)
    assert index.top_k([0.0, 0.0, 0.0], 5) == []
    index.close()


def test_single_vector(tmp_path) -> None:
    index = HNSWIndex(tmp_path / "hnsw", dim=2)
    index.insert([1.0, 2.0], RID(4, 2))

    assert index.top_k([1.0, 2.0], 3) == [(RID(4, 2), 1.0)]
    index.close()


def test_neighbor_lists_respect_capacity_and_are_symmetric_enough(tmp_path) -> None:
    items = clustered_vectors(300, DIM)
    index = _build(tmp_path / "hnsw", items, m=4, ef_construction=32)

    for node in range(len(index)):
        assert len(index.neighbors(node, 0)) <= 8
        assert node not in index.neighbors(node, 0)
        for layer in range(1, index.level(node) + 1):
            assert len(index.neighbors(node, layer)) <= 4
    assert index.max_level >= 1
    index.close()


def test_every_node_is_reachable_with_large_ef(tmp_path) -> None:
    items = clustered_vectors(200, DIM, seed=2)
    index = _build(tmp_path / "hnsw", items, m=4, ef_construction=32)

    result = index.top_k(items[0][0], 500, ef=500)

    assert sorted(rid for rid, _ in result) == sorted(rid for _, rid in items)
    index.close()


def test_query_vector_comes_first_with_max_score(tmp_path) -> None:
    items = clustered_vectors(500, DIM, seed=3)
    index = _build(tmp_path / "hnsw", items, m=8, ef_construction=64)

    for position in (0, 17, 250, 499):
        (rid, best), *_ = index.top_k(items[position][0], 5)
        assert rid == items[position][1]
        assert best == pytest.approx(1.0, abs=1e-5)
    index.close()


def test_persistence_after_close_and_reopen(tmp_path) -> None:
    items = clustered_vectors(300, DIM)
    path = tmp_path / "hnsw"
    index = _build(path, items, m=6, ef_construction=40, metric=COSINE, seed=7)
    query = items[42][0]
    before = index.top_k(query, 10)
    max_level = index.max_level
    index.close()

    reopened = HNSWIndex(path)
    assert (reopened.dim, reopened.metric, reopened.max_level, len(reopened)) == (
        DIM,
        COSINE,
        max_level,
        300,
    )
    assert reopened.top_k(query, 10) == before

    reopened.insert(query, RID(999, 0))
    assert RID(999, 0) in {rid for rid, _ in reopened.top_k(query, 2)}
    reopened.close()


def test_construction_is_deterministic_for_a_seed(tmp_path) -> None:
    items = clustered_vectors(150, DIM)
    a = _build(tmp_path / "a", items, m=4, seed=1)
    b = _build(tmp_path / "b", items, m=4, seed=1)

    assert [a.neighbors(n, 0) for n in range(150)] == [b.neighbors(n, 0) for n in range(150)]
    a.close()
    b.close()


def test_invalid_parameters_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        HNSWIndex(tmp_path / "x", dim=2, m=1)
    with pytest.raises(ValueError):
        HNSWIndex(tmp_path / "y", dim=2, metric="manhattan")


@pytest.mark.parametrize("metric", [EUCLIDEAN, COSINE])
def test_recall_at_10_against_brute_force(tmp_path, metric) -> None:
    items = clustered_vectors(2000, DIM, clusters=20, spread=0.3, seed=9)
    index = _build(tmp_path / "hnsw", items, m=8, ef_construction=64, metric=metric)
    oracle = FakeRankedIndex(vector_scorer)
    for vector, rid in items:
        oracle.add(vector, rid)
    rng = random.Random(5)
    queries = [[v + rng.gauss(0, 0.1) for v in items[rng.randrange(2000)][0]] for _ in range(30)]

    recalls = []
    for query in queries:
        found = [rid for rid, _ in index.top_k(query, 10, ef=64)]
        expected = [rid for rid, _ in oracle.top_k(query, 10, metric=metric)]
        recalls.append(recall_at_k(found, expected))
    index.close()

    assert sum(recalls) / len(recalls) >= 0.9
