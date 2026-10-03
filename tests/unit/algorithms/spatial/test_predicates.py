import pytest

from engine.algorithms.spatial import (
    Polygon2D,
    point_in_polygon,
    point_on_segment,
)
from engine.indexes.rtree import Point


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


def test_point_on_segment() -> None:
    assert point_on_segment(
        Point(2, 0),
        Point(0, 0),
        Point(4, 0),
    )


def test_point_outside_segment_extension() -> None:
    assert not point_on_segment(
        Point(5, 0),
        Point(0, 0),
        Point(4, 0),
    )


def test_point_inside_polygon(square: Polygon2D) -> None:
    assert point_in_polygon(Point(2, 2), square)


def test_point_outside_polygon(square: Polygon2D) -> None:
    assert not point_in_polygon(Point(5, 2), square)


def test_polygon_boundary_is_included(square: Polygon2D) -> None:
    assert point_in_polygon(Point(0, 2), square)


def test_polygon_vertex_is_included(square: Polygon2D) -> None:
    assert point_in_polygon(Point(4, 4), square)


def test_polygon_boundary_can_be_excluded(square: Polygon2D) -> None:
    assert not point_in_polygon(
        Point(0, 2),
        square,
        include_boundary=False,
    )


def test_polygon_orientation_does_not_change_result() -> None:
    clockwise = Polygon2D(
        (
            Point(0, 0),
            Point(0, 4),
            Point(4, 4),
            Point(4, 0),
        )
    )
    counterclockwise = Polygon2D(
        (
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(0, 4),
        )
    )

    point = Point(2, 2)

    assert point_in_polygon(point, clockwise)
    assert point_in_polygon(point, counterclockwise)


def test_concave_polygon_rejects_point_in_notch() -> None:
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

    assert point_in_polygon(Point(0.5, 3), polygon)
    assert not point_in_polygon(Point(3, 3), polygon)
