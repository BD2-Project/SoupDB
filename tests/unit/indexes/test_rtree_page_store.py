import random
import struct

import pytest

from engine.common.rid import RID
from engine.indexes.rtree import _page_codec
from engine.indexes.rtree._page_store import (
    METADATA_PAGE_ID,
    RTreeMetadata,
    RTreePageStore,
    decode_metadata,
    encode_metadata,
)
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.rtree import RTree
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 4096
FULL_RANGE = ((-1000, -1000), (1000, 1000))
RANGES = [((-100, -100), (100, 100)), ((0, -500), (500, 0)), FULL_RANGE]


def _open(path, capacity: int = 8) -> tuple[DiskManager, BufferManager, RTreePageStore]:
    disk = DiskManager(path, PAGE_SIZE)
    buffer = BufferManager(disk, capacity)
    return disk, buffer, RTreePageStore(disk, buffer)


def _persist(path, tree: RTree, capacity: int = 8) -> DiskManager:
    disk, _buffer, store = _open(path, capacity)
    store.save_tree(tree)
    disk.close()
    return disk


def _reopen(path, capacity: int = 8) -> RTree:
    disk, _buffer, store = _open(path, capacity)
    tree = store.load_tree()
    disk.close()
    return tree


def _saved_root(path) -> RTreeNode | None:
    disk, _buffer, store = _open(path)
    _order, root = store.load()
    disk.close()
    return root


def _build(order: int, points: list[tuple[int, int]]) -> RTree:
    tree = RTree(order=order)
    for i, point in enumerate(points):
        tree.insert(point, RID(i, i % 7))
    return tree


def _node_count(node: RTreeNode | None) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + sum(_node_count(child) for _mbr, child in node.entries)


def _random_points(count: int, seed: int = 7) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(-500, 500), rng.randint(-500, 500)) for _ in range(count)]


def _assert_same_contents(original: RTree, loaded: RTree, points) -> None:
    assert loaded.order == original.order
    for point in set(points):
        assert sorted(loaded.search(point)) == sorted(original.search(point))
    for lo, hi in RANGES:
        assert sorted(loaded.range_search(lo, hi)) == sorted(original.range_search(lo, hi))


def test_metadata_roundtrip() -> None:
    for metadata in (RTreeMetadata(order=4, root_page_id=-1), RTreeMetadata(102, 2**40)):
        page = encode_metadata(PAGE_SIZE, metadata)
        assert len(page) == PAGE_SIZE
        assert page[:16] == struct.pack(
            ">4sBBHq", b"RTM1", 1, 0, metadata.order, metadata.root_page_id
        )
        assert page[16:] == bytes(PAGE_SIZE - 16)
        assert decode_metadata(page) == metadata


def test_empty_tree_roundtrip(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    disk = _persist(path, RTree(order=5))
    assert disk.page_count == 1
    assert _saved_root(path) is None

    loaded = _reopen(path)
    assert loaded.order == 5
    assert loaded.search((1, 1)) == []
    assert loaded.range_search(*FULL_RANGE) == []


def test_load_without_metadata_rejected(tmp_path) -> None:
    disk, _buffer, store = _open(tmp_path / "rtree.db")
    with pytest.raises(ValueError, match="metadata"):
        store.load_tree()
    disk.close()


def test_single_leaf_roundtrip(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = [(1, 2), (3, 4), (-5, 6)]
    tree = _build(4, points)
    disk = _persist(path, tree)
    assert disk.page_count == 2
    assert _saved_root(path).is_leaf

    _assert_same_contents(tree, _reopen(path), points)


def test_root_split_roundtrip(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = [(0, 0), (10, 10), (20, 20)]
    tree = _build(2, points)
    _persist(path, tree)

    root = _saved_root(path)
    assert not root.is_leaf
    assert len(root.entries) == 2
    _assert_same_contents(tree, _reopen(path), points)


def test_multinode_tree_uses_one_page_per_node(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = _random_points(300)
    tree = _build(4, points)
    disk = _persist(path, tree)

    nodes = _node_count(_saved_root(path))
    assert nodes > 10
    assert disk.page_count == nodes + 1
    assert path.stat().st_size == (nodes + 1) * PAGE_SIZE

    loaded = _reopen(path)
    _assert_same_contents(tree, loaded, points)
    for i, point in enumerate(points):
        assert RID(i, i % 7) in loaded.search(point)


def test_loaded_internal_entries_reference_child_mbr(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(3, _random_points(80)))

    def check(node: RTreeNode) -> None:
        if node.is_leaf:
            return
        for mbr, child in node.entries:
            assert isinstance(child, RTreeNode)
            assert mbr == child.mbr
            check(child)

    check(_saved_root(path))


def test_range_search_preserved_after_reopen(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = _random_points(200, seed=3)
    tree = _build(5, points)
    _persist(path, tree)
    loaded = _reopen(path)

    for lo, hi in RANGES:
        assert sorted(loaded.range_search(lo, hi)) == sorted(tree.range_search(lo, hi))
    assert len(loaded.range_search(*FULL_RANGE)) == len(points)


def test_full_lifecycle_with_reopen_between_saves(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    first = _random_points(80, seed=1)
    second = _random_points(120, seed=2)
    everything = first + second
    expected = {}
    for i, point in enumerate(everything):
        expected.setdefault(point, []).append(RID(i, i % 7))

    tree = RTree(order=3)
    for i, point in enumerate(first):
        tree.insert(point, RID(i, i % 7))
    disk, buffer, store = _open(path)
    store.save_tree(tree)
    buffer.flush_all()
    disk.close()
    assert _node_count(_saved_root(path)) > 10

    disk, buffer, store = _open(path)
    tree = store.load_tree()
    for i, point in enumerate(first):
        assert RID(i, i % 7) in tree.search(point)
    assert len(tree.range_search(*FULL_RANGE)) == len(first)

    for i, point in enumerate(second, start=len(first)):
        tree.insert(point, RID(i, i % 7))
    store.save_tree(tree)
    disk.close()

    disk, _buffer, store = _open(path)
    reloaded = store.load_tree()
    disk.close()
    for point, rids in expected.items():
        assert sorted(reloaded.search(point)) == sorted(rids)
    assert sorted(reloaded.range_search(*FULL_RANGE)) == sorted(
        rid for rids in expected.values() for rid in rids
    )
    for lo, hi in RANGES:
        assert sorted(reloaded.range_search(lo, hi)) == sorted(tree.range_search(lo, hi))


def test_tree_api_delegates_to_low_level_api(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rtree.db"
    tree = _build(4, _random_points(40, seed=12))
    disk, _buffer, store = _open(path)
    calls = []
    save, load = store.save, store.load

    def spy_save(order, root):
        calls.append(("save", order, root))
        return save(order, root)

    def spy_load():
        result = load()
        calls.append(("load", *result))
        return result

    monkeypatch.setattr(store, "save", spy_save)
    monkeypatch.setattr(store, "load", spy_load)

    store.save_tree(tree)
    loaded = store.load_tree()
    disk.close()

    assert [call[:2] for call in calls] == [("save", 4), ("load", 4)]
    saved_root, loaded_root = calls[0][2], calls[1][2]
    assert _node_count(saved_root) == _node_count(loaded_root) > 1
    _assert_same_contents(tree, loaded, _random_points(40, seed=12))


def test_resave_with_same_store_reuses_pages(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = _random_points(120, seed=4)
    tree = _build(4, points[:60])
    disk, _buffer, store = _open(path)

    store.save_tree(tree)
    first_count = disk.page_count
    assert first_count == _node_count(_saved_root(path)) + 1

    store.save_tree(tree)
    assert disk.page_count == first_count

    for i, point in enumerate(points[60:], start=60):
        tree.insert(point, RID(i, i % 7))
    store.save_tree(tree)
    assert disk.page_count == _node_count(_saved_root(path)) + 1
    disk.close()

    _assert_same_contents(tree, _reopen(path), points)


def test_load_tree_keeps_page_reuse_for_existing_nodes(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(4, _random_points(80, seed=5)))

    disk, _buffer, store = _open(path)
    tree = store.load_tree()
    pages_after_load = disk.page_count
    store.save_tree(tree)
    assert disk.page_count == pages_after_load

    for i, point in enumerate(_random_points(80, seed=6)):
        tree.insert(point, RID(1000 + i, 0))
    store.save_tree(tree)
    assert disk.page_count == _node_count(_saved_root(path)) + 1
    disk.close()


def test_removed_nodes_leave_orphan_pages(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = _random_points(100, seed=8)
    tree = _build(3, points)
    disk, _buffer, store = _open(path)
    store.save_tree(tree)
    pages_before = disk.page_count

    for point in points[:80]:
        tree.remove(point)
    store.save_tree(tree)
    assert disk.page_count == pages_before
    assert disk.page_count > _node_count(_saved_root(path)) + 1
    disk.close()

    _assert_same_contents(tree, _reopen(path), points)


def test_buffer_capacity_one_forces_evictions(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = _random_points(250, seed=9)
    tree = _build(4, points)
    disk = _persist(path, tree, capacity=1)
    assert disk.writes >= disk.page_count

    loaded = _reopen(path, capacity=1)
    _assert_same_contents(tree, loaded, points)


def test_disk_manager_counts_reads_and_writes(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    tree = _build(4, _random_points(150, seed=10))

    disk = _persist(path, tree)
    nodes = _node_count(_saved_root(path))
    assert disk.writes >= nodes + 1

    disk, _buffer, store = _open(path)
    reads_before = disk.reads
    store.load_tree()
    assert disk.reads - reads_before == nodes + 1
    disk.close()


def _rewrite_page(path, page_id: int, data: bytes) -> None:
    disk = DiskManager(path, PAGE_SIZE)
    disk.write_page(page_id, data)
    disk.close()


def _read_page(path, page_id: int) -> bytes:
    disk = DiskManager(path, PAGE_SIZE)
    data = disk.read_page(page_id)
    disk.close()
    return data


def _load_error(path) -> None:
    disk, _buffer, store = _open(path)
    try:
        store.load()
    finally:
        disk.close()


@pytest.mark.parametrize(
    ("offset", "raw", "message"),
    [
        (0, b"XXXX", "magic"),
        (4, bytes((9,)), "version"),
        (5, bytes((1,)), "reserved"),
    ],
)
def test_corrupt_metadata_rejected(tmp_path, offset: int, raw: bytes, message: str) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(4, _random_points(30)))
    page = _read_page(path, METADATA_PAGE_ID)
    _rewrite_page(path, METADATA_PAGE_ID, page[:offset] + raw + page[offset + len(raw) :])

    with pytest.raises(ValueError, match=message):
        _load_error(path)


@pytest.mark.parametrize("root_page_id", [-2, 0, 99])
def test_invalid_root_page_id_rejected(tmp_path, root_page_id: int) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(4, _random_points(30)))
    _rewrite_page(
        path, METADATA_PAGE_ID, encode_metadata(PAGE_SIZE, RTreeMetadata(4, root_page_id))
    )

    with pytest.raises(ValueError, match="root_page_id"):
        _load_error(path)


@pytest.mark.parametrize("child_page_id", [0, 99])
def test_invalid_child_page_id_rejected(tmp_path, child_page_id: int) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(2, _random_points(10)))
    root_page_id = decode_metadata(_read_page(path, METADATA_PAGE_ID)).root_page_id
    root = _page_codec.decode_page(_read_page(path, root_page_id), PAGE_SIZE)
    (mbr, _child), *rest = root.entries
    corrupted = _page_codec.DecodedRTreePage(
        is_leaf=False, max_entries=root.max_entries, entries=((mbr, child_page_id), *rest)
    )
    _rewrite_page(path, root_page_id, _page_codec.encode_page(PAGE_SIZE, corrupted))

    with pytest.raises(ValueError, match="child_page_id"):
        _load_error(path)


def test_cyclic_page_reference_rejected(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(2, _random_points(10)))
    root_page_id = decode_metadata(_read_page(path, METADATA_PAGE_ID)).root_page_id
    root = _page_codec.decode_page(_read_page(path, root_page_id), PAGE_SIZE)
    (mbr, _child), *rest = root.entries
    corrupted = _page_codec.DecodedRTreePage(
        is_leaf=False, max_entries=root.max_entries, entries=((mbr, root_page_id), *rest)
    )
    _rewrite_page(path, root_page_id, _page_codec.encode_page(PAGE_SIZE, corrupted))

    with pytest.raises(ValueError, match="child_page_id"):
        _load_error(path)


def test_stale_child_mbr_rejected(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(2, _random_points(10)))
    root_page_id = decode_metadata(_read_page(path, METADATA_PAGE_ID)).root_page_id
    root = _page_codec.decode_page(_read_page(path, root_page_id), PAGE_SIZE)
    (mbr, child), *rest = root.entries
    wrong = type(mbr)(mbr.min_x - 1, mbr.min_y, mbr.max_x, mbr.max_y)
    corrupted = _page_codec.DecodedRTreePage(
        is_leaf=False, max_entries=root.max_entries, entries=((wrong, child), *rest)
    )
    _rewrite_page(path, root_page_id, _page_codec.encode_page(PAGE_SIZE, corrupted))

    with pytest.raises(ValueError, match="MBR"):
        _load_error(path)


@pytest.mark.parametrize("order", [1, 103, 500])
def test_order_incompatible_with_page_size_rejected_on_save(tmp_path, order: int) -> None:
    disk, _buffer, store = _open(tmp_path / "rtree.db")
    with pytest.raises(ValueError, match="order"):
        store.save(order, None)
    assert disk.page_count == 0
    disk.close()


def test_order_incompatible_with_page_size_rejected_on_load(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, RTree(order=4))
    _rewrite_page(path, METADATA_PAGE_ID, encode_metadata(PAGE_SIZE, RTreeMetadata(103, -1)))

    with pytest.raises(ValueError, match="order"):
        _load_error(path)


def test_max_order_for_page_size_roundtrip(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    points = _random_points(400, seed=11)
    tree = _build(102, points)
    _persist(path, tree)
    assert not _saved_root(path).is_leaf
    _assert_same_contents(tree, _reopen(path), points)


def test_node_order_mismatch_rejected(tmp_path) -> None:
    path = tmp_path / "rtree.db"
    _persist(path, _build(4, [(1, 1)]))
    _rewrite_page(path, METADATA_PAGE_ID, encode_metadata(PAGE_SIZE, RTreeMetadata(5, 1)))

    with pytest.raises(ValueError, match="max_entries"):
        _load_error(path)
