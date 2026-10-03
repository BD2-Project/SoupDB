from __future__ import annotations

import math

from engine.indexes.rtree import Point

from .validation import validate_cartesian_point


def euclidean_distance(a: Point, b: Point) -> float:
    a = validate_cartesian_point(a, name="a")
    b = validate_cartesian_point(b, name="b")

    return math.hypot(a.x - b.x, a.y - b.y)
