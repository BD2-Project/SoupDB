from __future__ import annotations

from engine.algorithms.spatial.metrics import SpatialMetric
from engine.algorithms.spatial.range_query import (
    iter_radius_entries,
    iter_range_entries,
)
from engine.algorithms.spatial.traversal import iter_leaf_entries
from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point
from engine.indexes.rtree.rtree import RTree


class SpatialQueries:
    def __init__(self, tree: RTree) -> None:
        if not isinstance(tree, RTree):
            raise TypeError("tree must be an RTree")

        self._tree = tree

    def _root(self) -> RTreeNode | None:
        self._tree._ensure_open()
        return self._tree._root

    def entries(self) -> tuple[tuple[Point, RID], ...]:
        return tuple(iter_leaf_entries(self._root()))

    def range_search(self, box: MBR) -> list[RID]:
        return [
            rid
            for _point, rid in iter_range_entries(
                self._root(),
                box,
            )
        ]

    def radius_search(
        self,
        center: Point,
        radius: float,
        metric: SpatialMetric,
    ) -> list[RID]:
        return [
            rid
            for _point, rid in iter_radius_entries(
                self._root(),
                center,
                radius,
                metric,
            )
        ]
