import pytest

from engine.algorithms.spatial import Polygon2D
from engine.indexes.rtree import MBR, Point


def test_polygon_accepts_three_vertices() -> None:
    polygon = Polygon2D(
        (
            Point(0, 0),
            Point(4, 0),
            Point(0, 4),
        )
    )

    assert len(polygon.vertices) == 3


def test_polygon_removes_repeated_closing_vertex() -> None:
    polygon = Polygon2D(
        (
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(0, 4),
            Point(0, 0),
        )
    )

    assert len(polygon.vertices) == 4


def test_polygon_converts_coordinates_to_float() -> None:
    polygon = Polygon2D(
        (
            Point(0, 0),
            Point(4, 0),
            Point(0, 4),
        )
    )

    assert polygon.vertices[0] == Point(0.0, 0.0)


def test_polygon_bounds() -> None:
    polygon = Polygon2D(
        (
            Point(-2, 1),
            Point(4, -3),
            Point(3, 5),
        )
    )

    assert polygon.bounds == MBR(-2.0, -3.0, 4.0, 5.0)


@pytest.mark.parametrize(
    "vertices",
    [
        (),
        (Point(0, 0),),
        (Point(0, 0), Point(1, 1)),
    ],
)
def test_polygon_rejects_too_few_vertices(
    vertices: tuple[Point, ...],
) -> None:
    with pytest.raises(ValueError):
        Polygon2D(vertices)


def test_polygon_rejects_duplicate_vertices() -> None:
    with pytest.raises(ValueError):
        Polygon2D(
            (
                Point(0, 0),
                Point(4, 0),
                Point(4, 4),
                Point(4, 0),
            )
        )


def test_polygon_rejects_collinear_vertices() -> None:
    with pytest.raises(ValueError):
        Polygon2D(
            (
                Point(0, 0),
                Point(1, 1),
                Point(2, 2),
            )
        )


def test_polygon_rejects_self_intersection() -> None:
    with pytest.raises(ValueError):
        Polygon2D(
            (
                Point(0, 0),
                Point(4, 4),
                Point(0, 4),
                Point(4, 0),
            )
        )
