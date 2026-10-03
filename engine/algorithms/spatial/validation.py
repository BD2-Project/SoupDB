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
