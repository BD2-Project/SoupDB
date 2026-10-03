from __future__ import annotations

import math

from engine.indexes.rtree import MBR, Point

from .validation import (
    validate_cartesian_point,
    validate_earth_radius,
    validate_geographic_point,
)


def euclidean_min_distance(center: Point, box: MBR) -> float:
    center = validate_cartesian_point(center, name="center")

    dx = max(box.min_x - center.x, 0.0, center.x - box.max_x)
    dy = max(box.min_y - center.y, 0.0, center.y - box.max_y)

    return math.hypot(dx, dy)


def haversine_latitude_lower_bound(
    center: Point,
    box: MBR,
    *,
    earth_radius_m: float,
) -> float:
    center = validate_geographic_point(center, name="center")
    radius = validate_earth_radius(earth_radius_m)

    if center.y < box.min_y:
        latitude_gap = box.min_y - center.y
    elif center.y > box.max_y:
        latitude_gap = center.y - box.max_y
    else:
        latitude_gap = 0.0

    return radius * math.radians(latitude_gap)
