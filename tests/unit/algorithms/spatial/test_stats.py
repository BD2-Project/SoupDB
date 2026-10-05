from collections import Counter

import pytest

from engine.algorithms.spatial import (
    Polygon2D,
    SpatialQueryStats,
    euclidean_metric,
)
from engine.common.rid import RID
from engine.indexes.rtree import MBR, Point, RTree, SpatialQueries
from engine.indexes.rtree.node import new_internal, new_leaf


def test_spatial_query_stats_start_at_zero() -> None:
    stats = SpatialQueryStats()

    assert stats.nodes_visited == 0
    assert stats.distance_evaluations == 0
    assert stats.result_count == 0


def test_spatial_query_stats_reset() -> None:
    stats = SpatialQueryStats(
        nodes_visited=3,
        nodes_pruned=2,
        result_count=5,
    )

    stats.reset()

    assert stats == SpatialQueryStats()


def test_range_stats_do_not_change_results() -> None:
    tree = RTree(order=2)

    for index in range(6):
        tree.insert(Point(index, index), RID(0, index))

    queries = SpatialQueries(tree)
    box = MBR(1, 1, 4, 4)

    expected = queries.range_search(box)

    stats = SpatialQueryStats()
    actual = queries.range_search(box, stats=stats)

    assert Counter(actual) == Counter(expected)
    assert stats.result_count == len(actual)
    assert stats.nodes_visited == (stats.internal_nodes_visited + stats.leaf_nodes_visited)
    assert stats.entries_examined >= stats.result_count


def test_stats_are_reset_between_queries() -> None:
    tree = RTree(order=2)

    for index in range(6):
        tree.insert(Point(index, index), RID(0, index))

    queries = SpatialQueries(tree)
    stats = SpatialQueryStats()

    queries.range_search(
        MBR(0, 0, 5, 5),
        stats=stats,
    )

    assert stats.result_count == 6

    queries.range_search(
        MBR(0, 0, 0, 0),
        stats=stats,
    )

    assert stats.result_count == 1


def test_radius_stats_count_distances_and_pruning() -> None:
    near_leaf = new_leaf(
        4,
        [
            (Point(0, 0), RID(0, 0)),
            (Point(1, 0), RID(0, 1)),
        ],
    )
    far_leaf = new_leaf(
        4,
        [
            (Point(100, 100), RID(0, 2)),
            (Point(101, 100), RID(0, 3)),
        ],
    )

    assert near_leaf.mbr is not None
    assert far_leaf.mbr is not None

    root = new_internal(
        4,
        [
            (near_leaf.mbr, near_leaf),
            (far_leaf.mbr, far_leaf),
        ],
    )

    tree = RTree(order=4)
    tree._root = root

    stats = SpatialQueryStats()

    result = SpatialQueries(tree).radius_search(
        Point(0, 0),
        2.0,
        euclidean_metric(),
        stats=stats,
    )

    assert result == [RID(0, 0), RID(0, 1)]
    assert stats.distance_evaluations == 2
    assert stats.entries_examined == 2
    assert stats.nodes_pruned >= 1
    assert stats.bound_evaluations >= 3
    assert stats.result_count == 2


def test_knn_stats_track_frontier_and_candidates() -> None:
    near_leaf = new_leaf(
        4,
        [
            (Point(0, 0), RID(0, 0)),
            (Point(1, 0), RID(0, 1)),
        ],
    )
    far_leaf = new_leaf(
        4,
        [
            (Point(100, 100), RID(0, 2)),
            (Point(101, 100), RID(0, 3)),
        ],
    )

    assert near_leaf.mbr is not None
    assert far_leaf.mbr is not None

    root = new_internal(
        4,
        [
            (near_leaf.mbr, near_leaf),
            (far_leaf.mbr, far_leaf),
        ],
    )

    tree = RTree(order=4)
    tree._root = root

    stats = SpatialQueryStats()

    result = SpatialQueries(tree).knn(
        Point(0, 0),
        2,
        euclidean_metric(),
        stats=stats,
    )

    assert result == [RID(0, 0), RID(0, 1)]
    assert stats.distance_evaluations == 2
    assert stats.candidates_peak <= 2
    assert stats.candidates_peak == 2
    assert stats.heap_pushes >= 1
    assert stats.heap_pops >= 1
    assert stats.frontier_peak >= 1
    assert stats.nodes_pruned >= 1
    assert stats.result_count == 2


def test_polygon_stats_count_exact_predicates() -> None:
    polygon = Polygon2D(
        (
            Point(0, 0),
            Point(3, 0),
            Point(3, 3),
            Point(0, 3),
        )
    )

    tree = RTree(order=2)
    tree.insert(Point(1, 1), RID(0, 0))
    tree.insert(Point(2, 2), RID(0, 1))
    tree.insert(Point(100, 100), RID(0, 2))

    stats = SpatialQueryStats()

    result = SpatialQueries(tree).polygon_search(
        polygon,
        stats=stats,
    )

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])
    assert stats.polygon_tests == 2
    assert stats.result_count == 2
    assert stats.entries_examined >= stats.polygon_tests


def test_queries_reject_invalid_stats() -> None:
    with pytest.raises(
        TypeError,
        match="stats must be a SpatialQueryStats",
    ):
        SpatialQueries(RTree()).range_search(
            MBR(0, 0, 1, 1),
            stats=object(),  # type: ignore[arg-type]
        )
