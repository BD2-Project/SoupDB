from __future__ import annotations

from collections.abc import Iterator

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point

from .metrics import SpatialMetric
from .traversal import iter_child_nodes
from .validation import validate_mbr, validate_radius


def iter_range_entries(
    root: RTreeNode | None,
    box: MBR,
) -> Iterator[tuple[Point, RID]]:
    box = validate_mbr(box)

    if root is None or root.mbr is None:
        return

    if not root.mbr.intersects(box):
        return

    stack = [root]

    while stack:
        node = stack.pop()

        if node.is_leaf:
            for point, rid in node.entries:
                if box.contains(point):
                    yield point, rid
            continue

        children = [
            child for child_mbr, child in iter_child_nodes(node) if child_mbr.intersects(box)
        ]
        stack.extend(reversed(children))


def iter_radius_entries(
    root: RTreeNode | None,
    center: Point,
    radius: float,
    metric: SpatialMetric,
) -> Iterator[tuple[Point, RID]]:
    if not isinstance(metric, SpatialMetric):
        raise TypeError("metric must be a SpatialMetric")

    radius = validate_radius(radius)

    if root is None or root.mbr is None:
        return

    if metric.lower_bound(center, root.mbr) > radius:
        return

    stack = [root]

    while stack:
        node = stack.pop()

        if node.is_leaf:
            for point, rid in node.entries:
                if metric.distance(center, point) <= radius:
                    yield point, rid
            continue

        children = [
            child
            for child_mbr, child in iter_child_nodes(node)
            if metric.lower_bound(center, child_mbr) <= radius
        ]
        stack.extend(reversed(children))
