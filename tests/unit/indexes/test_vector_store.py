"""Paged vector store and brute-force k-NN."""

import random
import struct

import pytest

from engine.common.rid import RID
from engine.indexes.vector.brute_force import BruteForceIndex
from engine.indexes.vector.metrics import COSINE, EUCLIDEAN, INNER_PRODUCT, score
from engine.indexes.vector.store import VectorStore, max_dim
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 512


def _f32(values: list[float]) -> tuple[float, ...]:
    return struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values))


@pytest.fixture
def store(tmp_path):
    vectors = VectorStore.open(
        tmp_path / "vectors.db", dim=4, page_size=PAGE_SIZE, buffer_capacity=2
    )
    yield vectors
    vectors.close()


def test_read_returns_exactly_what_was_written(store) -> None:
    vector = [0.5, -1.25, 3.0, 1e-3]

    vid = store.append(vector, RID(7, 3))

    rid, stored, stored_norm = store.get(vid)
    assert rid == RID(7, 3)
    assert stored == _f32(vector)
    assert stored_norm == pytest.approx(sum(v * v for v in _f32(vector)) ** 0.5)


def test_vids_are_dense_and_span_pages(store) -> None:
    vectors = [[float(i), float(-i), i / 2, 1.0] for i in range(100)]

    vids = [store.append(vector, RID(i, 0)) for i, vector in enumerate(vectors)]

    assert vids == list(range(100))
    assert store.disk_manager.page_count > 2
    assert [store.vector(vid)[0] for vid in vids] == [_f32(v) for v in vectors]
    assert [rid for _vid, rid, _v, _n in store.scan()] == [RID(i, 0) for i in range(100)]


def test_wrong_dimension_is_rejected(store) -> None:
    with pytest.raises(ValueError):
        store.append([1.0, 2.0], RID(0, 0))


def test_store_survives_close_and_reopen(tmp_path) -> None:
    path = tmp_path / "vectors.db"
    store = VectorStore.open(path, dim=3, page_size=PAGE_SIZE)
    for i in range(50):
        store.append([i, i + 0.5, -i], RID(i, i))
    store.close()

    reopened = VectorStore.open(path, page_size=PAGE_SIZE)
    try:
        assert reopened.dim == 3
        assert len(reopened) == 50
        rid, vector, _norm = reopened.get(49)
        assert (rid, vector) == (RID(49, 49), _f32([49, 49.5, -49]))
    finally:
        reopened.close()


def test_reopen_with_other_dim_is_rejected(tmp_path) -> None:
    path = tmp_path / "vectors.db"
    VectorStore.open(path, dim=3, page_size=PAGE_SIZE).close()

    with pytest.raises(ValueError):
        VectorStore.open(path, dim=5, page_size=PAGE_SIZE)


def test_dimension_larger_than_a_page_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        VectorStore.open(tmp_path / "v.db", dim=max_dim(PAGE_SIZE) + 1, page_size=PAGE_SIZE)


def test_store_goes_through_the_disk_manager(tmp_path) -> None:
    disk = DiskManager(tmp_path / "v.db", page_size=PAGE_SIZE)
    store = VectorStore(disk, BufferManager(disk, capacity=1), dim=8)
    for i in range(40):
        store.append([float(i)] * 8, RID(i, 0))
    store.flush()
    writes = disk.writes
    reads = disk.reads

    list(store.scan())

    assert disk.reads > reads
    assert disk.writes == writes
    disk.close()


@pytest.mark.parametrize("metric", [EUCLIDEAN, COSINE, INNER_PRODUCT])
def test_brute_force_matches_sorted_scores(tmp_path, metric) -> None:
    rng = random.Random(3)
    vectors = [_f32([rng.uniform(-1, 1) for _ in range(6)]) for _ in range(200)]
    index = BruteForceIndex.open(tmp_path / "bf.db", dim=6, page_size=PAGE_SIZE, buffer_capacity=4)
    for i, vector in enumerate(vectors):
        index.insert(vector, RID(i, 0))
    query = [rng.uniform(-1, 1) for _ in range(6)]

    result = index.top_k(query, 10, metric=metric)
    index.close()

    expected = sorted(
        ((RID(i, 0), score(metric, query, v)) for i, v in enumerate(vectors)), key=lambda p: -p[1]
    )[:10]
    assert [rid for rid, _ in result] == [rid for rid, _ in expected]
    assert [s for _, s in result] == pytest.approx([s for _, s in expected])


def test_brute_force_finds_the_query_vector_first(tmp_path) -> None:
    index = BruteForceIndex.open(tmp_path / "bf.db", dim=2, page_size=PAGE_SIZE)
    for i in range(20):
        index.insert([float(i), float(i % 3)], RID(i, 0))

    (rid, best), *_ = index.top_k([7.0, 1.0], 3)
    index.close()

    assert rid == RID(7, 0)
    assert best == 1.0
