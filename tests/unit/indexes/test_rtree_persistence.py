"""Tests de persistencia del R-Tree (save/load/open) y fuzz contra oráculo."""

from collections import defaultdict

from hypothesis import given
from hypothesis import strategies as st

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point
from engine.indexes.rtree.rtree import RTree


def test_save_load_roundtrip(tmp_path) -> None:
    path = tmp_path / "rtree.bin"
    tree = RTree(order=2)
    points = [(x, x) for x in range(20)]
    for i, point in enumerate(points):
        tree.insert(point, RID(0, i))
    tree.save(path)

    loaded = RTree.load(path)
    assert loaded.order == 2
    for i, point in enumerate(points):
        assert loaded.search(point) == [RID(0, i)]


def test_open_creates_and_reopens(tmp_path) -> None:
    path = tmp_path / "rtree.bin"
    tree = RTree.open(path, order=3)
    tree.insert((10, 10), RID(1, 1))
    tree.insert((20, 5), RID(1, 2))
    tree.save(path)

    reopened = RTree.open(path)
    assert reopened.order == 3
    assert reopened.search((10, 10)) == [RID(1, 1)]
    assert reopened.search((20, 5)) == [RID(1, 2)]


def test_persistence_preserves_mbrs(tmp_path) -> None:
    path = tmp_path / "rtree.bin"
    tree = RTree(order=2)
    for point in [(0, 0), (1, 1), (5, 5), (9, 9)]:
        tree.insert(point, RID(0, 0))
    tree.save(path)
    loaded = RTree.load(path)
    assert loaded._root.mbr == tree._root.mbr


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        ),
        max_size=120,
        unique_by=lambda p: p,
    )
)
def test_fuzz_search_against_oracle(points) -> None:
    tree = RTree(order=2)
    oracle: dict[tuple[int, int], list[RID]] = defaultdict(list)
    for i, point in enumerate(points):
        tree.insert(point, RID(0, i))
        oracle[point].append(RID(0, i))

    for point, expected in oracle.items():
        assert sorted(tree.search(point)) == sorted(expected)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        ),
        max_size=120,
        unique_by=lambda p: p,
    )
)
def test_fuzz_range_against_oracle(points) -> None:
    tree = RTree(order=2)
    oracle: dict[tuple[int, int], list[RID]] = defaultdict(list)
    for i, point in enumerate(points):
        tree.insert(point, RID(0, i))
        oracle[point].append(RID(0, i))

    lo, hi = Point(25, 25), Point(75, 75)
    box = MBR(25, 25, 75, 75)
    expected = [
        rid
        for point, rids in oracle.items()
        if box.contains(Point(point[0], point[1]))
        for rid in rids
    ]
    assert sorted(tree.range_search(lo, hi)) == sorted(expected)
