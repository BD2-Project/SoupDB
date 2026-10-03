from __future__ import annotations

import math

from engine.indexes.rtree import Point

from .validation import (
    validate_cartesian_point,
    validate_earth_radius,
    validate_geographic_point,
)

EARTH_RADIUS_M = 6_371_000.0


def euclidean_distance(a: Point, b: Point) -> float:
    a = validate_cartesian_point(a, name="a")
    b = validate_cartesian_point(b, name="b")

    return math.hypot(a.x - b.x, a.y - b.y)


def haversine_distance(
    a: Point,
    b: Point,
    *,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> float:
    a = validate_geographic_point(a, name="a")
    b = validate_geographic_point(b, name="b")
    radius = validate_earth_radius(earth_radius_m)

    lat_a = math.radians(a.y)
    lat_b = math.radians(b.y)

    delta_lat = lat_b - lat_a
    delta_lon = math.radians(b.x - a.x)
    delta_lon = (delta_lon + math.pi) % math.tau - math.pi

    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))

    central_angle = 2.0 * math.atan2(
        math.sqrt(haversine),
        math.sqrt(1.0 - haversine),
    )

    return radius * central_angle
