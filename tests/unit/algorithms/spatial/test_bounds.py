import math

import pytest

from engine.algorithms.spatial import (
    EARTH_RADIUS_M,
    euclidean_distance,
    euclidean_metric,
    euclidean_min_distance,
    haversine_distance,
    haversine_latitude_lower_bound,
    haversine_metric,
)
from engine.indexes.rtree import MBR, Point


def test_euclidean_bound_is_zero_inside_box() -> None:
    box = MBR(0.0, 0.0, 4.0, 4.0)

    assert euclidean_min_distance(Point(2.0, 2.0), box) == pytest.approx(0.0)


def test_euclidean_bound_is_zero_on_box_boundary() -> None:
    box = MBR(0.0, 0.0, 4.0, 4.0)

    assert euclidean_min_distance(Point(4.0, 2.0), box) == pytest.approx(0.0)


def test_euclidean_bound_outside_one_axis() -> None:
    box = MBR(0.0, 0.0, 4.0, 4.0)

    assert euclidean_min_distance(Point(5.0, 2.0), box) == pytest.approx(1.0)


def test_euclidean_bound_outside_two_axes() -> None:
    box = MBR(0.0, 0.0, 4.0, 4.0)

    assert euclidean_min_distance(Point(7.0, 8.0), box) == pytest.approx(5.0)


def test_euclidean_bound_for_point_box_matches_exact_distance() -> None:
    point = Point(3.0, 4.0)
    box = MBR(3.0, 4.0, 3.0, 4.0)
    center = Point(0.0, 0.0)

    assert euclidean_min_distance(center, box) == pytest.approx(euclidean_distance(center, point))


@pytest.mark.parametrize(
    ("center", "contained"),
    [
        (Point(5.0, 2.0), Point(4.0, 1.0)),
        (Point(7.0, 8.0), Point(4.0, 4.0)),
        (Point(-3.0, -3.0), Point(0.0, 0.0)),
    ],
)
def test_euclidean_bound_does_not_exceed_distance_to_contained_point(
    center: Point,
    contained: Point,
) -> None:
    box = MBR(0.0, 0.0, 4.0, 4.0)

    assert euclidean_min_distance(center, box) <= euclidean_distance(
        center,
        contained,
    )


def test_haversine_latitude_bound_is_zero_inside_latitude_band() -> None:
    box = MBR(-80.0, -13.0, -70.0, -11.0)

    bound = haversine_latitude_lower_bound(
        Point(-120.0, -12.0),
        box,
        earth_radius_m=EARTH_RADIUS_M,
    )

    assert bound == pytest.approx(0.0)


def test_haversine_latitude_bound_for_one_degree_gap() -> None:
    box = MBR(-80.0, -11.0, -70.0, -10.0)

    bound = haversine_latitude_lower_bound(
        Point(-77.0, -12.0),
        box,
        earth_radius_m=EARTH_RADIUS_M,
    )

    assert bound == pytest.approx(EARTH_RADIUS_M * math.pi / 180.0)


def test_haversine_bound_does_not_exceed_distance_to_contained_point() -> None:
    center = Point(-77.0, -12.0)
    contained = Point(-76.0, -10.0)
    box = MBR(-78.0, -10.0, -75.0, -8.0)

    bound = haversine_latitude_lower_bound(
        center,
        box,
        earth_radius_m=EARTH_RADIUS_M,
    )

    assert bound <= haversine_distance(center, contained)


def test_euclidean_metric_pairs_distance_and_bound() -> None:
    metric = euclidean_metric()
    center = Point(0.0, 0.0)
    point = Point(3.0, 4.0)
    box = MBR(3.0, 4.0, 3.0, 4.0)

    assert metric.name == "euclidean"
    assert metric.distance(center, point) == pytest.approx(5.0)
    assert metric.lower_bound(center, box) == pytest.approx(5.0)


def test_haversine_metric_uses_same_earth_radius_for_distance_and_bound() -> None:
    radius = 1_000_000.0
    metric = haversine_metric(earth_radius_m=radius)

    center = Point(0.0, 0.0)
    point = Point(0.0, 1.0)
    box = MBR(0.0, 1.0, 0.0, 1.0)

    expected = radius * math.pi / 180.0

    assert metric.name == "haversine"
    assert metric.unit == "meters"
    assert metric.distance(center, point) == pytest.approx(expected)
    assert metric.lower_bound(center, box) == pytest.approx(expected)
