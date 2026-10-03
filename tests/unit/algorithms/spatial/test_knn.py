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
from engine.indexes.rtree import Point, RTree, SpatialQueries
from engine.indexes.rtree.node import new_internal, new_leaf


def test_knn_empty_tree() -> None:
    result = SpatialQueries(RTree()).knn(
        Point(0, 0),
        3,
        euclidean_metric(),
    )

    assert result == []


def test_knn_returns_nearest_point() -> None:
    tree = RTree(order=2)
    tree.insert(Point(5, 5), RID(0, 0))
    tree.insert(Point(1, 0), RID(0, 1))
    tree.insert(Point(3, 4), RID(0, 2))

    result = SpatialQueries(tree).knn(
        Point(0, 0),
        1,
        euclidean_metric(),
    )

    assert result == [RID(0, 1)]


def test_knn_k_larger_than_collection_returns_all() -> None:
    tree = RTree(order=2)
    tree.insert(Point(3, 4), RID(0, 0))
    tree.insert(Point(1, 0), RID(0, 1))
    tree.insert(Point(2, 0), RID(0, 2))

    result = SpatialQueries(tree).knn(
        Point(0, 0),
        10,
        euclidean_metric(),
    )

    assert result == [
        RID(0, 1),
        RID(0, 2),
        RID(0, 0),
    ]


@pytest.mark.parametrize("k", [0, -1])
def test_knn_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError):
        SpatialQueries(RTree()).knn(
            Point(0, 0),
            k,
            euclidean_metric(),
        )


@pytest.mark.parametrize("k", [True, 1.5, "2"])
def test_knn_rejects_invalid_k_type(k: object) -> None:
    with pytest.raises(TypeError):
        SpatialQueries(RTree()).knn(
            Point(0, 0),
            k,  # type: ignore[arg-type]
            euclidean_metric(),
        )


def test_knn_preserves_duplicate_entries() -> None:
    tree = RTree(order=2)

    for _ in range(3):
        tree.insert(Point(1, 1), RID(0, 0))

    result = SpatialQueries(tree).knn(
        Point(0, 0),
        3,
        euclidean_metric(),
    )

    assert result == [
        RID(0, 0),
        RID(0, 0),
        RID(0, 0),
    ]


def test_knn_matches_baseline_tie_breaking() -> None:
    entries = [
        (Point(1, 0), RID(0, 4)),
        (Point(0, 1), RID(0, 3)),
        (Point(0, -1), RID(0, 2)),
        (Point(-1, 0), RID(0, 1)),
        (Point(0, 0), RID(0, 0)),
        (Point(3, 4), RID(0, 5)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    center = Point(0, 0)

    expected = SequentialSpatialScan(entries).knn(
        center,
        3,
        euclidean_distance,
    )
    actual = SpatialQueries(tree).knn(
        center,
        3,
        euclidean_metric(),
    )

    assert actual == expected


def test_knn_matches_sequential_baseline_euclidean() -> None:
    entries = [
        (Point(-4, 1), RID(0, 0)),
        (Point(1, 1), RID(0, 1)),
        (Point(2, 2), RID(0, 2)),
        (Point(3, 4), RID(0, 3)),
        (Point(10, 10), RID(0, 4)),
        (Point(-1, 0), RID(0, 5)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    center = Point(0, 0)

    expected = SequentialSpatialScan(entries).knn(
        center,
        4,
        euclidean_distance,
    )
    actual = SpatialQueries(tree).knn(
        center,
        4,
        euclidean_metric(),
    )

    assert actual == expected


def test_knn_matches_sequential_baseline_haversine() -> None:
    entries = [
        (Point(-77.0428, -12.0464), RID(0, 0)),
        (Point(-77.0430, -12.0500), RID(0, 1)),
        (Point(-77.0300, -12.0600), RID(0, 2)),
        (Point(-77.1000, -12.1000), RID(0, 3)),
        (Point(-76.9500, -12.0000), RID(0, 4)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    center = Point(-77.0428, -12.0464)

    expected = SequentialSpatialScan(entries).knn(
        center,
        3,
        haversine_distance,
    )
    actual = SpatialQueries(tree).knn(
        center,
        3,
        haversine_metric(),
    )

    assert actual == expected


def test_knn_hits_returns_point_rid_and_distance() -> None:
    tree = RTree(order=2)
    tree.insert(Point(3, 4), RID(0, 0))
    tree.insert(Point(1, 0), RID(0, 1))

    hits = SpatialQueries(tree).knn_hits(
        Point(0, 0),
        2,
        euclidean_metric(),
    )

    assert hits[0].point == Point(1, 0)
    assert hits[0].rid == RID(0, 1)
    assert hits[0].distance == pytest.approx(1.0)

    assert hits[1].point == Point(3, 4)
    assert hits[1].rid == RID(0, 0)
    assert hits[1].distance == pytest.approx(5.0)


def test_knn_explores_lower_bound_equal_to_tau() -> None:
    worse_leaf = new_leaf(
        2,
        [(Point(1, 0), RID(0, 1))],
    )
    better_leaf = new_leaf(
        2,
        [(Point(-1, 0), RID(0, 2))],
    )

    assert worse_leaf.mbr is not None
    assert better_leaf.mbr is not None

    root = new_internal(
        2,
        [
            (worse_leaf.mbr, worse_leaf),
            (better_leaf.mbr, better_leaf),
        ],
    )

    tree = RTree(order=2)
    tree._root = root

    result = SpatialQueries(tree).knn(
        Point(0, 0),
        1,
        euclidean_metric(),
    )

    assert result == [RID(0, 2)]


def test_knn_prunes_distant_subtree() -> None:
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

    result = SpatialQueries(tree).knn(
        Point(0, 0),
        2,
        metric,
    )

    assert result == [RID(0, 0), RID(0, 1)]
    assert distance_calls == 2


def test_knn_rejects_invalid_metric() -> None:
    with pytest.raises(TypeError, match="metric must be a SpatialMetric"):
        SpatialQueries(RTree()).knn(
            Point(0, 0),
            1,
            object(),  # type: ignore[arg-type]
        )
