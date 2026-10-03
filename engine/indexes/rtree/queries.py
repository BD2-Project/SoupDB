from __future__ import annotations

from engine.algorithms.spatial.geometry import Polygon2D
from engine.algorithms.spatial.knn import (
    SpatialHit,
    knn_search,
)
from engine.algorithms.spatial.knn import (
    knn_hits as search_knn_hits,
)
from engine.algorithms.spatial.metrics import SpatialMetric
from engine.algorithms.spatial.polygon_query import (
    polygon_search as search_polygon,
)
from engine.algorithms.spatial.range_query import (
    iter_radius_entries,
    iter_range_entries,
)
from engine.algorithms.spatial.stats import SpatialQueryStats
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

    @staticmethod
    def _prepare_stats(
        stats: SpatialQueryStats | None,
    ) -> SpatialQueryStats | None:
        if stats is None:
            return None

        if not isinstance(stats, SpatialQueryStats):
            raise TypeError("stats must be a SpatialQueryStats")

        stats.reset()

        return stats

    @staticmethod
    def _finish_stats(
        stats: SpatialQueryStats | None,
        result_count: int,
    ) -> None:
        if stats is not None:
            stats.result_count = result_count

    def entries(self) -> tuple[tuple[Point, RID], ...]:
        return tuple(iter_leaf_entries(self._root()))

    def range_search(
        self,
        box: MBR,
        *,
        stats: SpatialQueryStats | None = None,
    ) -> list[RID]:
        stats = self._prepare_stats(stats)

        result = [
            rid
            for _point, rid in iter_range_entries(
                self._root(),
                box,
                stats=stats,
            )
        ]

        self._finish_stats(stats, len(result))

        return result

    def radius_search(
        self,
        center: Point,
        radius: float,
        metric: SpatialMetric,
        *,
        stats: SpatialQueryStats | None = None,
    ) -> list[RID]:
        stats = self._prepare_stats(stats)

        result = [
            rid
            for _point, rid in iter_radius_entries(
                self._root(),
                center,
                radius,
                metric,
                stats=stats,
            )
        ]

        self._finish_stats(stats, len(result))

        return result

    def knn(
        self,
        center: Point,
        k: int,
        metric: SpatialMetric,
        *,
        stats: SpatialQueryStats | None = None,
    ) -> list[RID]:
        stats = self._prepare_stats(stats)

        result = knn_search(
            self._root(),
            center,
            k,
            metric,
            stats=stats,
        )

        self._finish_stats(stats, len(result))

        return result

    def knn_hits(
        self,
        center: Point,
        k: int,
        metric: SpatialMetric,
        *,
        stats: SpatialQueryStats | None = None,
    ) -> list[SpatialHit]:
        stats = self._prepare_stats(stats)

        result = search_knn_hits(
            self._root(),
            center,
            k,
            metric,
            stats=stats,
        )

        self._finish_stats(stats, len(result))

        return result

    def polygon_search(
        self,
        polygon: Polygon2D,
        *,
        stats: SpatialQueryStats | None = None,
    ) -> list[RID]:
        stats = self._prepare_stats(stats)

        result = search_polygon(
            self._root(),
            polygon,
            stats=stats,
        )

        self._finish_stats(stats, len(result))

        return result
