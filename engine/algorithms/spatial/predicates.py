from __future__ import annotations

from engine.indexes.rtree.point import Point

from .geometry import Polygon2D
from .validation import validate_cartesian_point


def point_on_segment(point: Point, a: Point, b: Point) -> bool:
    point = validate_cartesian_point(point, name="point")
    a = validate_cartesian_point(a, name="a")
    b = validate_cartesian_point(b, name="b")

    cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)

    if cross != 0.0:
        return False

    return min(a.x, b.x) <= point.x <= max(a.x, b.x) and min(a.y, b.y) <= point.y <= max(a.y, b.y)


def point_in_polygon(
    point: Point,
    polygon: Polygon2D,
    *,
    include_boundary: bool = True,
) -> bool:
    point = validate_cartesian_point(point, name="point")

    if not isinstance(polygon, Polygon2D):
        raise TypeError("polygon must be a Polygon2D")

    if not polygon.bounds.contains(point):
        return False

    vertices = polygon.vertices
    inside = False

    for index, current in enumerate(vertices):
        following = vertices[(index + 1) % len(vertices)]

        if point_on_segment(point, current, following):
            return include_boundary

        crosses_ray = (current.y > point.y) != (following.y > point.y)

        if not crosses_ray:
            continue

        intersection_x = current.x + (
            (point.y - current.y) * (following.x - current.x) / (following.y - current.y)
        )

        if point.x < intersection_x:
            inside = not inside

    return inside
