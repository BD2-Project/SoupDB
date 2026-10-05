import math

import pytest

from engine.algorithms.spatial import (
    EARTH_RADIUS_M,
    euclidean_distance,
    haversine_distance,
    point_from_latlon,
    validate_cartesian_point,
    validate_geographic_point,
)
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


def test_point_from_latlon_converts_to_internal_order() -> None:
    point = point_from_latlon(-12.0464, -77.0428)

    assert point == Point(-77.0428, -12.0464)


def test_haversine_same_point_is_zero() -> None:
    point = Point(-77.0428, -12.0464)

    assert haversine_distance(point, point) == pytest.approx(0.0)


def test_haversine_one_degree_at_equator() -> None:
    distance = haversine_distance(Point(0.0, 0.0), Point(1.0, 0.0))

    expected = EARTH_RADIUS_M * math.pi / 180.0

    assert distance == pytest.approx(expected)


def test_haversine_equator_to_pole() -> None:
    distance = haversine_distance(Point(0.0, 0.0), Point(0.0, 90.0))

    assert distance == pytest.approx(EARTH_RADIUS_M * math.pi / 2.0)


def test_haversine_antipodal_points() -> None:
    distance = haversine_distance(Point(0.0, 0.0), Point(180.0, 0.0))

    assert distance == pytest.approx(EARTH_RADIUS_M * math.pi)


def test_haversine_handles_antimeridian() -> None:
    distance = haversine_distance(
        Point(179.9, 0.0),
        Point(-179.9, 0.0),
    )

    expected = EARTH_RADIUS_M * math.radians(0.2)

    assert distance == pytest.approx(expected)


def test_haversine_is_symmetric() -> None:
    a = Point(-77.0428, -12.0464)
    b = Point(-77.0300, -12.1000)

    assert haversine_distance(a, b) == pytest.approx(haversine_distance(b, a))


@pytest.mark.parametrize(
    "point",
    [
        Point(180.1, 0.0),
        Point(-180.1, 0.0),
        Point(0.0, 90.1),
        Point(0.0, -90.1),
    ],
)
def test_validate_geographic_point_rejects_out_of_range_coordinates(
    point: Point,
) -> None:
    with pytest.raises(ValueError):
        validate_geographic_point(point)


@pytest.mark.parametrize(
    "radius",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_haversine_rejects_invalid_earth_radius(radius: float) -> None:
    with pytest.raises(ValueError):
        haversine_distance(
            Point(0.0, 0.0),
            Point(1.0, 1.0),
            earth_radius_m=radius,
        )
