from __future__ import annotations

from collections.abc import Iterator

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point

from .metrics import SpatialMetric
from .stats import SpatialQueryStats
from .traversal import iter_child_nodes
from .validation import validate_mbr, validate_radius


def iter_range_entries(
    root: RTreeNode | None,
    box: MBR,
    *,
    stats: SpatialQueryStats | None = None,
) -> Iterator[tuple[Point, RID]]:
    box = validate_mbr(box)

    if root is None or root.mbr is None:
        return

    if stats is not None:
        stats.mbr_tests += 1

    if not root.mbr.intersects(box):
        if stats is not None:
            stats.nodes_pruned += 1
        return

    stack = [root]

    while stack:
        node = stack.pop()

        if stats is not None:
            stats.visit_node(is_leaf=node.is_leaf)

        if node.is_leaf:
            for point, rid in node.entries:
                if stats is not None:
                    stats.entries_examined += 1

                if box.contains(point):
                    yield point, rid

            continue

        children = []

        for child_mbr, child in iter_child_nodes(node):
            if stats is not None:
                stats.mbr_tests += 1

            if child_mbr.intersects(box):
                children.append(child)
            elif stats is not None:
                stats.nodes_pruned += 1

        stack.extend(reversed(children))


def iter_radius_entries(
    root: RTreeNode | None,
    center: Point,
    radius: float,
    metric: SpatialMetric,
    *,
    stats: SpatialQueryStats | None = None,
) -> Iterator[tuple[Point, RID]]:
    if not isinstance(metric, SpatialMetric):
        raise TypeError("metric must be a SpatialMetric")

    radius = validate_radius(radius)

    if root is None or root.mbr is None:
        return

    if stats is not None:
        stats.bound_evaluations += 1

    if metric.lower_bound(center, root.mbr) > radius:
        if stats is not None:
            stats.nodes_pruned += 1
        return

    stack = [root]

    while stack:
        node = stack.pop()

        if stats is not None:
            stats.visit_node(is_leaf=node.is_leaf)

        if node.is_leaf:
            for point, rid in node.entries:
                if stats is not None:
                    stats.entries_examined += 1
                    stats.distance_evaluations += 1

                if metric.distance(center, point) <= radius:
                    yield point, rid

            continue

        children = []

        for child_mbr, child in iter_child_nodes(node):
            if stats is not None:
                stats.bound_evaluations += 1

            if metric.lower_bound(center, child_mbr) <= radius:
                children.append(child)
            elif stats is not None:
                stats.nodes_pruned += 1

        stack.extend(reversed(children))
