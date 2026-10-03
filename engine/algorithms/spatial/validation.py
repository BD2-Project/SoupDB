from __future__ import annotations

import math

from engine.indexes.rtree import Point


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number


def validate_cartesian_point(point: Point, *, name: str = "point") -> Point:
    if not isinstance(point, Point):
        raise TypeError(f"{name} must be a Point")

    return Point(
        _finite_number(point.x, name=f"{name}.x"),
        _finite_number(point.y, name=f"{name}.y"),
    )


def validate_geographic_point(point: Point, *, name: str = "point") -> Point:
    point = validate_cartesian_point(point, name=name)

    if not -180.0 <= point.x <= 180.0:
        raise ValueError(f"{name}.x longitude must be between -180 and 180")

    if not -90.0 <= point.y <= 90.0:
        raise ValueError(f"{name}.y latitude must be between -90 and 90")

    return point


def validate_earth_radius(
    earth_radius_m: object,
    *,
    name: str = "earth_radius_m",
) -> float:
    radius = _finite_number(earth_radius_m, name=name)

    if radius <= 0:
        raise ValueError(f"{name} must be positive")

    return radius


def point_from_latlon(latitude: object, longitude: object) -> Point:
    point = Point(
        _finite_number(longitude, name="longitude"),
        _finite_number(latitude, name="latitude"),
    )

    return validate_geographic_point(point)
