from __future__ import annotations

from collections.abc import Iterator

from engine.common.rid import RID
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point

from .geometry import Polygon2D
from .predicates import point_in_polygon
from .range_query import iter_range_entries
from .stats import SpatialQueryStats


def iter_polygon_entries(
    root: RTreeNode | None,
    polygon: Polygon2D,
    *,
    stats: SpatialQueryStats | None = None,
) -> Iterator[tuple[Point, RID]]:
    if not isinstance(polygon, Polygon2D):
        raise TypeError("polygon must be a Polygon2D")

    for point, rid in iter_range_entries(
        root,
        polygon.bounds,
        stats=stats,
    ):
        if stats is not None:
            stats.polygon_tests += 1

        if point_in_polygon(point, polygon):
            yield point, rid


def polygon_search(
    root: RTreeNode | None,
    polygon: Polygon2D,
    *,
    stats: SpatialQueryStats | None = None,
) -> list[RID]:
    return [
        rid
        for _point, rid in iter_polygon_entries(
            root,
            polygon,
            stats=stats,
        )
    ]
