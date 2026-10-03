import math
from collections import Counter

import pytest

from benchmarks.spatial_baseline import SequentialSpatialScan
from engine.common.rid import RID
from engine.indexes.rtree import MBR, Point, RTree, SpatialQueries


def test_range_search_empty_tree() -> None:
    queries = SpatialQueries(RTree())

    assert queries.range_search(MBR(0, 0, 4, 4)) == []


def test_range_search_matches_existing_rtree_api() -> None:
    tree = RTree(order=2)

    entries = [
        (Point(0, 0), RID(0, 0)),
        (Point(1, 1), RID(0, 1)),
        (Point(2, 2), RID(0, 2)),
        (Point(3, 3), RID(0, 3)),
        (Point(4, 4), RID(0, 4)),
        (Point(5, 5), RID(0, 5)),
    ]

    for point, rid in entries:
        tree.insert(point, rid)

    box = MBR(1, 1, 4, 4)

    expected = tree.range_search(
        Point(box.min_x, box.min_y),
        Point(box.max_x, box.max_y),
    )
    actual = SpatialQueries(tree).range_search(box)

    assert Counter(actual) == Counter(expected)


def test_range_search_matches_sequential_baseline() -> None:
    entries = [
        (Point(0, 0), RID(0, 0)),
        (Point(1, 1), RID(0, 1)),
        (Point(2, 2), RID(0, 2)),
        (Point(3, 3), RID(0, 3)),
        (Point(4, 4), RID(0, 4)),
        (Point(5, 5), RID(0, 5)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    box = MBR(1, 1, 4, 4)

    expected = SequentialSpatialScan(entries).range_search(box)
    actual = SpatialQueries(tree).range_search(box)

    assert Counter(actual) == Counter(expected)


def test_range_search_includes_boundaries() -> None:
    tree = RTree(order=2)
    tree.insert(Point(0, 0), RID(0, 0))
    tree.insert(Point(4, 4), RID(0, 1))
    tree.insert(Point(2, 2), RID(0, 2))
    tree.insert(Point(5, 5), RID(0, 3))

    result = SpatialQueries(tree).range_search(MBR(0, 0, 4, 4))

    assert Counter(result) == Counter(
        [
            RID(0, 0),
            RID(0, 1),
            RID(0, 2),
        ]
    )


def test_range_search_supports_degenerate_point_box() -> None:
    tree = RTree(order=2)
    tree.insert(Point(2, 2), RID(0, 0))
    tree.insert(Point(2, 2), RID(0, 1))
    tree.insert(Point(3, 2), RID(0, 2))

    result = SpatialQueries(tree).range_search(MBR(2, 2, 2, 2))

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])


def test_range_search_preserves_duplicate_coordinates() -> None:
    tree = RTree(order=2)
    tree.insert(Point(1, 1), RID(0, 0))
    tree.insert(Point(1, 1), RID(0, 1))

    result = SpatialQueries(tree).range_search(MBR(0, 0, 2, 2))

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])


@pytest.mark.parametrize(
    "box",
    [
        MBR(5, 0, 4, 1),
        MBR(0, 5, 1, 4),
    ],
)
def test_range_search_rejects_inverted_mbr(box: MBR) -> None:
    with pytest.raises(ValueError):
        SpatialQueries(RTree()).range_search(box)


@pytest.mark.parametrize(
    "box",
    [
        MBR(math.nan, 0, 1, 1),
        MBR(0, 0, math.inf, 1),
    ],
)
def test_range_search_rejects_non_finite_mbr(box: MBR) -> None:
    with pytest.raises(ValueError):
        SpatialQueries(RTree()).range_search(box)
