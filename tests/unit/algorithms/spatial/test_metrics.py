import math

import pytest

from engine.algorithms.spatial import euclidean_distance, validate_cartesian_point
from engine.indexes.rtree import Point


def test_euclidean_distance_345_triangle() -> None:
    assert euclidean_distance(Point(0, 0), Point(3, 4)) == pytest.approx(5.0)


def test_euclidean_distance_same_point_is_zero() -> None:
    point = Point(2.5, -7.25)

    assert euclidean_distance(point, point) == pytest.approx(0.0)


def test_euclidean_distance_is_symmetric() -> None:
    a = Point(-3.5, 2.0)
    b = Point(4.25, -6.0)

    assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))


def test_euclidean_distance_supports_negative_coordinates() -> None:
    assert euclidean_distance(Point(-3, -4), Point(0, 0)) == pytest.approx(5.0)


def test_validate_cartesian_point_converts_coordinates_to_float() -> None:
    assert validate_cartesian_point(Point(3, -4)) == Point(3.0, -4.0)


@pytest.mark.parametrize(
    "point",
    [
        Point(math.nan, 0.0),
        Point(math.inf, 0.0),
        Point(0.0, -math.inf),
    ],
)
def test_validate_cartesian_point_rejects_non_finite_coordinates(
    point: Point,
) -> None:
    with pytest.raises(ValueError):
        validate_cartesian_point(point)


@pytest.mark.parametrize(
    "point",
    [
        Point(True, 0.0),
        Point(0.0, False),
        Point("1", 0.0),
    ],
)
def test_validate_cartesian_point_rejects_invalid_coordinate_types(
    point: Point,
) -> None:
    with pytest.raises(TypeError):
        validate_cartesian_point(point)


def test_validate_cartesian_point_rejects_non_point() -> None:
    with pytest.raises(TypeError):
        validate_cartesian_point((1.0, 2.0))  # type: ignore[arg-type]
