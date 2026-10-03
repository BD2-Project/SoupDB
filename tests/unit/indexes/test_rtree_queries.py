from copy import deepcopy

import pytest

from engine.algorithms.spatial.traversal import iter_child_nodes
from engine.common.rid import RID
from engine.indexes.rtree import MBR, Point, RTree, SpatialQueries


def test_spatial_queries_empty_tree_has_no_entries() -> None:
    queries = SpatialQueries(RTree())

    assert queries.entries() == ()


def test_spatial_queries_reads_leaf_entries() -> None:
    tree = RTree(order=4)
    tree.insert(Point(1, 2), RID(0, 1))
    tree.insert(Point(3, 4), RID(0, 2))

    queries = SpatialQueries(tree)

    assert queries.entries() == (
        (Point(1.0, 2.0), RID(0, 1)),
        (Point(3.0, 4.0), RID(0, 2)),
    )


def test_spatial_queries_reads_entries_across_internal_nodes() -> None:
    tree = RTree(order=2)

    expected = []
    for index in range(6):
        point = Point(float(index), float(index))
        rid = RID(0, index)
        tree.insert(point, rid)
        expected.append((point, rid))

    assert sorted(SpatialQueries(tree).entries()) == sorted(expected)


def test_iter_child_nodes_uses_current_child_mbr() -> None:
    tree = RTree(order=2)

    for index in range(5):
        tree.insert(Point(index, index), RID(0, index))

    root = tree._root

    assert root is not None
    assert not root.is_leaf

    _stored_mbr, child = root.entries[0]
    current_mbr = child.mbr

    assert current_mbr is not None

    root.entries[0] = (
        MBR(-999.0, -999.0, -998.0, -998.0),
        child,
    )

    children = list(iter_child_nodes(root))

    assert children[0][0] == current_mbr


def test_spatial_queries_do_not_mutate_tree() -> None:
    tree = RTree(order=2)

    for index in range(6):
        tree.insert(Point(index, index), RID(0, index))

    before = deepcopy(tree._root)

    SpatialQueries(tree).entries()

    assert tree._root == before


def test_spatial_queries_reject_closed_tree() -> None:
    tree = RTree()
    queries = SpatialQueries(tree)

    tree.close()

    with pytest.raises(RuntimeError, match="R-Tree is closed"):
        queries.entries()


def test_spatial_queries_reject_non_rtree() -> None:
    with pytest.raises(TypeError, match="tree must be an RTree"):
        SpatialQueries(object())  # type: ignore[arg-type]
