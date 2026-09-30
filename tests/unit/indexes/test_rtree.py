"""Tests del R-Tree: inserción, división, búsqueda, rango y borrado."""

import pytest

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point
from engine.indexes.rtree.rtree import RTree


def test_insert_and_search_point() -> None:
    tree = RTree(order=4)
    rid = RID(0, 1)
    tree.insert((1, 2), rid)
    assert tree.search((1, 2)) == [rid]
    assert tree.search((9, 9)) == []


def test_duplicate_points_return_all_rids() -> None:
    tree = RTree(order=2)
    rids = [RID(0, i) for i in range(4)]
    for rid in rids:
        tree.insert((5, 5), rid)
    assert sorted(tree.search((5, 5))) == sorted(rids)


def test_split_preserves_all_points() -> None:
    tree = RTree(order=2)
    points = [(x, x) for x in range(10)]
    for i, point in enumerate(points):
        tree.insert(point, RID(0, i))
    for i, point in enumerate(points):
        assert tree.search(point) == [RID(0, i)]


def test_range_search_box() -> None:
    tree = RTree(order=2)
    for i, point in enumerate([(1, 1), (2, 5), (5, 2), (8, 8), (10, 10)]):
        tree.insert(point, RID(0, i))
    found = {rid for point in [(1, 1), (2, 5), (5, 2)] for rid in tree.search(point)}
    assert sorted(tree.range_search(Point(0, 0), Point(6, 6))) == sorted([rid for rid in found])
    assert tree.search((8, 8)) == [RID(0, 3)]


def test_scalar_keys_support_1d_range() -> None:
    tree = RTree(order=2)
    for key, rid in [(10, RID(0, 1)), (20, RID(0, 2)), (30, RID(0, 3)), (40, RID(0, 4))]:
        tree.insert(key, rid)
    assert sorted(tree.range_search(20, 30)) == sorted([RID(0, 2), RID(0, 3)])


def test_root_mbr_covers_all_inserted() -> None:
    tree = RTree(order=2)
    for point in [(0, 0), (1, 1), (5, 5), (9, 9)]:
        tree.insert(point, RID(0, 0))
    assert tree._root.mbr == MBR(0, 0, 9, 9)


def test_remove_single_rid() -> None:
    tree = RTree(order=2)
    first, second = RID(1, 1), RID(1, 2)
    tree.insert(5, first)
    tree.insert(5, second)
    assert tree.remove(5, first) == 1
    assert tree.search(5) == [second]


def test_remove_all_under_key() -> None:
    tree = RTree(order=2)
    for i in range(3):
        tree.insert(7, RID(0, i))
    assert tree.remove(7) == 3
    assert tree.search(7) == []


def test_remove_missing_key_returns_zero() -> None:
    tree = RTree(order=4)
    assert tree.remove(999) == 0


def test_supports_range_is_true() -> None:
    assert RTree().supports_range is True


def test_invalid_order_rejected() -> None:
    with pytest.raises(ValueError):
        RTree(order=1)
