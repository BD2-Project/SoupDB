"""IVF index: training, inverted lists on disk, persistence and recall."""

import random

import pytest

from engine.common.rid import RID
from engine.indexes.vector.ivf import IVFIndex
from engine.indexes.vector.kmeans import kmeans, nearest, reservoir_sample
from engine.indexes.vector.metrics import COSINE, EUCLIDEAN, INNER_PRODUCT
from tests.fakes.fake_ranked_index import FakeRankedIndex, vector_scorer
from tests.unit.indexes.vector_fixtures import clustered_vectors, recall_at_k

PAGE_SIZE = 1024
DIM = 16


def _oracle(items) -> FakeRankedIndex:
    oracle = FakeRankedIndex(vector_scorer)
    for vector, rid in items:
        oracle.add(vector, rid)
    return oracle


def test_reservoir_sample_is_bounded_and_streaming() -> None:
    sample = reservoir_sample(([float(i)] for i in range(10_000)), 50, random.Random(0))

    assert len(sample) == 50
    assert len(set(sample)) == 50


def test_kmeans_separates_obvious_clusters() -> None:
    rng = random.Random(1)
    left = [(rng.gauss(-10, 0.1), rng.gauss(0, 0.1)) for _ in range(30)]
    right = [(rng.gauss(10, 0.1), rng.gauss(0, 0.1)) for _ in range(30)]

    centroids = kmeans(left + right, 2, iterations=10, rng=rng)

    assert sorted(round(c[0]) for c in centroids) == [-10, 10]
    assert nearest((-9.0, 0.0), centroids) != nearest((9.0, 0.0), centroids)


def test_kmeans_with_fewer_points_than_k() -> None:
    assert len(kmeans([(0.0,), (1.0,)], 5, iterations=3, rng=random.Random(0))) == 2


def test_insert_before_training_is_rejected(tmp_path) -> None:
    index = IVFIndex.open(tmp_path / "ivf.db", dim=2, page_size=PAGE_SIZE)
    with pytest.raises(ValueError):
        index.insert([0.0, 0.0], RID(0, 0))
    index.close()


def test_build_requires_a_reiterable_source(tmp_path) -> None:
    items = clustered_vectors(10, 2)
    with pytest.raises(ValueError):
        IVFIndex.build(tmp_path / "ivf.db", iter(items), dim=2, nlist=2)


def test_every_vector_lands_in_exactly_one_list(tmp_path) -> None:
    items = clustered_vectors(500, DIM)
    index = IVFIndex.build(tmp_path / "ivf.db", items, dim=DIM, nlist=8, page_size=PAGE_SIZE)

    assert len(index) == 500
    assert sum(index.list_sizes()) == 500
    assert index.top_k(items[0][0], 1000, nprobe=8)[0][0] == items[0][1]
    assert len(index.top_k(items[0][0], 1000, nprobe=8)) == 500
    index.close()


def test_probing_every_list_is_exact(tmp_path) -> None:
    items = clustered_vectors(300, DIM, seed=4)
    index = IVFIndex.build(tmp_path / "ivf.db", items, dim=DIM, nlist=6, page_size=PAGE_SIZE)
    oracle = _oracle(items)
    query = [0.2] * DIM

    for metric in (EUCLIDEAN, COSINE, INNER_PRODUCT):
        found = index.top_k(query, 10, metric=metric, nprobe=6)
        expected = oracle.top_k(query, 10, metric=metric)
        assert [s for _, s in found] == pytest.approx([s for _, s in expected], rel=1e-5)
    index.close()


def test_query_reads_only_the_probed_lists(tmp_path) -> None:
    items = clustered_vectors(2000, DIM)
    index = IVFIndex.build(
        tmp_path / "ivf.db", items, dim=DIM, nlist=16, page_size=PAGE_SIZE, buffer_capacity=4
    )
    disk = index.disk_manager

    before = disk.reads
    index.top_k(items[0][0], 10, nprobe=1)
    one_list = disk.reads - before
    before = disk.reads
    index.top_k(items[0][0], 10, nprobe=16)
    all_lists = disk.reads - before

    assert one_list < all_lists
    index.close()


def test_persistence_after_close_and_reopen(tmp_path) -> None:
    items = clustered_vectors(400, DIM)
    path = tmp_path / "ivf.db"
    index = IVFIndex.build(path, items, dim=DIM, nlist=8, page_size=PAGE_SIZE)
    query = items[123][0]
    before = index.top_k(query, 10, nprobe=3)
    sizes = index.list_sizes()
    index.close()

    reopened = IVFIndex.open(path, page_size=PAGE_SIZE)
    assert (reopened.dim, reopened.nlist, len(reopened)) == (DIM, 8, 400)
    assert reopened.list_sizes() == sizes
    assert reopened.top_k(query, 10, nprobe=3) == before

    # Se puede seguir insertando después de reabrir.
    reopened.insert(query, RID(999, 0))
    assert RID(999, 0) in {rid for rid, _ in reopened.top_k(query, 2, nprobe=1)}
    reopened.close()


@pytest.mark.parametrize("metric", [EUCLIDEAN, COSINE])
def test_recall_at_10_against_brute_force(tmp_path, metric) -> None:
    items = clustered_vectors(2000, DIM, clusters=20, spread=0.3, seed=9)
    index = IVFIndex.build(tmp_path / "ivf.db", items, dim=DIM, nlist=32, page_size=4096)
    oracle = _oracle(items)
    rng = random.Random(5)
    queries = [[v + rng.gauss(0, 0.1) for v in items[rng.randrange(2000)][0]] for _ in range(30)]

    recalls = []
    for query in queries:
        found = [rid for rid, _ in index.top_k(query, 10, metric=metric, nprobe=8)]
        expected = [rid for rid, _ in oracle.top_k(query, 10, metric=metric)]
        recalls.append(recall_at_k(found, expected))
    index.close()

    assert sum(recalls) / len(recalls) >= 0.9
