from collections import Counter

import pytest

import engine.algorithms.spatial.polygon_query as polygon_query_module
from engine.algorithms.spatial import Polygon2D, point_in_polygon
from engine.common.rid import RID
from engine.indexes.rtree import Point, RTree, SpatialQueries
from engine.indexes.rtree.node import new_internal, new_leaf


@pytest.fixture
def square() -> Polygon2D:
    return Polygon2D(
        (
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(0, 4),
        )
    )


def test_polygon_search_empty_tree(square: Polygon2D) -> None:
    result = SpatialQueries(RTree()).polygon_search(square)

    assert result == []


def test_polygon_search_matches_sequential_scan(
    square: Polygon2D,
) -> None:
    entries = [
        (Point(-1, 2), RID(0, 0)),
        (Point(0, 2), RID(0, 1)),
        (Point(2, 2), RID(0, 2)),
        (Point(4, 4), RID(0, 3)),
        (Point(5, 2), RID(0, 4)),
    ]

    tree = RTree(order=2)

    for point, rid in entries:
        tree.insert(point, rid)

    expected = [rid for point, rid in entries if point_in_polygon(point, square)]
    actual = SpatialQueries(tree).polygon_search(square)

    assert Counter(actual) == Counter(expected)


def test_polygon_search_includes_boundary_and_vertex(
    square: Polygon2D,
) -> None:
    tree = RTree(order=2)
    tree.insert(Point(0, 2), RID(0, 0))
    tree.insert(Point(4, 4), RID(0, 1))
    tree.insert(Point(5, 2), RID(0, 2))

    result = SpatialQueries(tree).polygon_search(square)

    assert Counter(result) == Counter(
        [
            RID(0, 0),
            RID(0, 1),
        ]
    )


def test_polygon_search_preserves_duplicate_entries(
    square: Polygon2D,
) -> None:
    tree = RTree(order=2)

    tree.insert(Point(2, 2), RID(0, 0))
    tree.insert(Point(2, 2), RID(0, 0))
    tree.insert(Point(2, 2), RID(0, 1))

    result = SpatialQueries(tree).polygon_search(square)

    assert Counter(result) == Counter(
        [
            RID(0, 0),
            RID(0, 0),
            RID(0, 1),
        ]
    )


def test_polygon_search_handles_concave_polygon() -> None:
    polygon = Polygon2D(
        (
            Point(0, 0),
            Point(4, 0),
            Point(4, 1),
            Point(1, 1),
            Point(1, 4),
            Point(0, 4),
        )
    )

    tree = RTree(order=2)
    tree.insert(Point(0.5, 3), RID(0, 0))
    tree.insert(Point(3, 3), RID(0, 1))
    tree.insert(Point(2, 0.5), RID(0, 2))

    result = SpatialQueries(tree).polygon_search(polygon)

    assert Counter(result) == Counter(
        [
            RID(0, 0),
            RID(0, 2),
        ]
    )


def test_polygon_search_prefilters_with_polygon_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    near_leaf = new_leaf(
        4,
        [
            (Point(1, 1), RID(0, 0)),
            (Point(2, 2), RID(0, 1)),
        ],
    )
    far_leaf = new_leaf(
        4,
        [
            (Point(100, 100), RID(0, 2)),
            (Point(101, 101), RID(0, 3)),
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

    polygon = Polygon2D(
        (
            Point(0, 0),
            Point(3, 0),
            Point(3, 3),
            Point(0, 3),
        )
    )

    predicate_calls = 0
    real_predicate = point_in_polygon

    def counting_predicate(
        point: Point,
        polygon: Polygon2D,
        *,
        include_boundary: bool = True,
    ) -> bool:
        nonlocal predicate_calls
        predicate_calls += 1

        return real_predicate(
            point,
            polygon,
            include_boundary=include_boundary,
        )

    monkeypatch.setattr(
        polygon_query_module,
        "point_in_polygon",
        counting_predicate,
    )

    result = SpatialQueries(tree).polygon_search(polygon)

    assert Counter(result) == Counter([RID(0, 0), RID(0, 1)])
    assert predicate_calls == 2


def test_polygon_search_rejects_invalid_polygon() -> None:
    with pytest.raises(TypeError, match="polygon must be a Polygon2D"):
        SpatialQueries(RTree()).polygon_search(
            object(),  # type: ignore[arg-type]
        )
