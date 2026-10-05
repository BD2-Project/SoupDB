from __future__ import annotations

from dataclasses import dataclass

from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point

from .validation import validate_cartesian_point


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _point_on_segment(point: Point, a: Point, b: Point) -> bool:
    return (
        _cross(a, b, point) == 0.0
        and min(a.x, b.x) <= point.x <= max(a.x, b.x)
        and min(a.y, b.y) <= point.y <= max(a.y, b.y)
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    ab_c = _cross(a, b, c)
    ab_d = _cross(a, b, d)
    cd_a = _cross(c, d, a)
    cd_b = _cross(c, d, b)

    if (ab_c > 0.0 and ab_d < 0.0 or ab_c < 0.0 and ab_d > 0.0) and (
        cd_a > 0.0 and cd_b < 0.0 or cd_a < 0.0 and cd_b > 0.0
    ):
        return True

    return (
        (ab_c == 0.0 and _point_on_segment(c, a, b))
        or (ab_d == 0.0 and _point_on_segment(d, a, b))
        or (cd_a == 0.0 and _point_on_segment(a, c, d))
        or (cd_b == 0.0 and _point_on_segment(b, c, d))
    )


@dataclass(frozen=True)
class Polygon2D:
    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        vertices = tuple(
            validate_cartesian_point(vertex, name="vertex") for vertex in self.vertices
        )

        if len(vertices) >= 2 and vertices[0] == vertices[-1]:
            vertices = vertices[:-1]

        if len(vertices) < 3:
            raise ValueError("polygon must contain at least three vertices")

        if len(set(vertices)) != len(vertices):
            raise ValueError("polygon vertices must be distinct")

        if self._signed_area(vertices) == 0.0:
            raise ValueError("polygon area must be non-zero")

        self._validate_simple(vertices)

        object.__setattr__(self, "vertices", vertices)

    @staticmethod
    def _signed_area(vertices: tuple[Point, ...]) -> float:
        area = 0.0

        for index, current in enumerate(vertices):
            following = vertices[(index + 1) % len(vertices)]
            area += current.x * following.y - following.x * current.y

        return area / 2.0

    @staticmethod
    def _validate_simple(vertices: tuple[Point, ...]) -> None:
        count = len(vertices)

        for first_index in range(count):
            a = vertices[first_index]
            b = vertices[(first_index + 1) % count]

            for second_index in range(first_index + 1, count):
                if second_index == first_index:
                    continue
                if second_index == (first_index + 1) % count:
                    continue
                if first_index == (second_index + 1) % count:
                    continue

                c = vertices[second_index]
                d = vertices[(second_index + 1) % count]

                if _segments_intersect(a, b, c, d):
                    raise ValueError("polygon must not self-intersect")

    @property
    def bounds(self) -> MBR:
        xs = [vertex.x for vertex in self.vertices]
        ys = [vertex.y for vertex in self.vertices]

        return MBR(
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )
