import math
from collections import Counter

import pytest

from benchmarks.spatial_baseline import SequentialSpatialScan
from engine.algorithms.spatial import (
    SpatialMetric,
    euclidean_distance,
    euclidean_metric,
    haversine_distance,
    haversine_metric,
)
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


def test_radius_search_empty_tree() -> None:
    result = SpatialQueries(RTree()).radius_search(
        Point(0, 0),
        5.0,
        euclidean_metric(),
    )

    assert result == []


def test_radius_search_zero_includes_exact_point() -> None:
    tree = RTree(order=2)
    tree.insert(Point(1, 1), RID(0, 0))
    tree.insert(Point(1, 1), RID(0, 1))
    tree.insert(Point(1, 2), RID(0, 2))

    result = SpatialQueries(tree).radius_search(
        Point(1, 1),
        0.0,
        euclidean_metric(),
    )

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])


def test_radius_search_includes_exact_boundary() -> None:
    tree = RTree(order=2)
    tree.insert(Point(3, 4), RID(0, 0))
    tree.insert(Point(6, 8), RID(0, 1))

    result = SpatialQueries(tree).radius_search(
        Point(0, 0),
        5.0,
        euclidean_metric(),
    )

    assert result == [RID(0, 0)]


def test_radius_search_excludes_square_corner_outside_circle() -> None:
    tree = RTree(order=2)
    tree.insert(Point(1, 0), RID(0, 0))
    tree.insert(Point(0, 1), RID(0, 1))
    tree.insert(Point(1, 1), RID(0, 2))

    result = SpatialQueries(tree).radius_search(
        Point(0, 0),
        1.0,
        euclidean_metric(),
    )

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])


def test_radius_search_matches_sequential_baseline_euclidean() -> None:
    entries = [
        (Point(0, 0), RID(0, 0)),
        (Point(1, 1), RID(0, 1)),
        (Point(3, 4), RID(0, 2)),
        (Point(5, 5), RID(0, 3)),
        (Point(-2, 0), RID(0, 4)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    center = Point(0, 0)
    radius = 5.0

    expected = SequentialSpatialScan(entries).radius_search(
        center,
        radius,
        euclidean_distance,
    )
    actual = SpatialQueries(tree).radius_search(
        center,
        radius,
        euclidean_metric(),
    )

    assert Counter(actual) == Counter(expected)


def test_radius_search_matches_sequential_baseline_haversine() -> None:
    entries = [
        (Point(-77.0428, -12.0464), RID(0, 0)),
        (Point(-77.0430, -12.0500), RID(0, 1)),
        (Point(-77.0300, -12.0600), RID(0, 2)),
        (Point(-77.1000, -12.1000), RID(0, 3)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    center = Point(-77.0428, -12.0464)
    radius = 5_000.0

    expected = SequentialSpatialScan(entries).radius_search(
        center,
        radius,
        haversine_distance,
    )
    actual = SpatialQueries(tree).radius_search(
        center,
        radius,
        haversine_metric(),
    )

    assert Counter(actual) == Counter(expected)


@pytest.mark.parametrize(
    "radius",
    [
        -1.0,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_radius_search_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError):
        SpatialQueries(RTree()).radius_search(
            Point(0, 0),
            radius,
            euclidean_metric(),
        )


def test_radius_search_rejects_invalid_metric() -> None:
    with pytest.raises(TypeError, match="metric must be a SpatialMetric"):
        SpatialQueries(RTree()).radius_search(
            Point(0, 0),
            1.0,
            object(),  # type: ignore[arg-type]
        )


def test_radius_search_prunes_distant_subtree() -> None:
    entries = [
        (Point(0, 0), RID(0, 0)),
        (Point(1, 0), RID(0, 1)),
        (Point(100, 100), RID(0, 2)),
        (Point(101, 100), RID(0, 3)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    distance_calls = 0
    base_metric = euclidean_metric()

    def counting_distance(a: Point, b: Point) -> float:
        nonlocal distance_calls
        distance_calls += 1
        return euclidean_distance(a, b)

    metric = SpatialMetric(
        name="counting-euclidean",
        unit="coordinate_units",
        distance=counting_distance,
        lower_bound=base_metric.lower_bound,
    )

    result = SpatialQueries(tree).radius_search(
        Point(0, 0),
        2.0,
        metric,
    )

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])
    assert distance_calls < len(entries)
