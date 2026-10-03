from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count

from engine.common.rid import RID
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point

from .metrics import SpatialMetric
from .traversal import iter_child_nodes
from .validation import validate_cartesian_point


@dataclass(frozen=True)
class SpatialHit:
    point: Point
    rid: RID
    distance: float


def _validate_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an int")

    if k <= 0:
        raise ValueError("k must be positive")

    return k


def _rank(hit: SpatialHit) -> tuple[float, float, float, int, int]:
    return (
        hit.distance,
        hit.point.x,
        hit.point.y,
        hit.rid.page_id,
        hit.rid.slot,
    )


def _negative_rank(
    rank: tuple[float, float, float, int, int],
) -> tuple[float, float, float, int, int]:
    distance, x, y, page_id, slot = rank

    return (
        -distance,
        -x,
        -y,
        -page_id,
        -slot,
    )


def _positive_rank(
    negative_rank: tuple[float, float, float, int, int],
) -> tuple[float, float, float, int, int]:
    distance, x, y, page_id, slot = negative_rank

    return (
        -distance,
        -x,
        -y,
        -page_id,
        -slot,
    )


def knn_hits(
    root: RTreeNode | None,
    center: Point,
    k: int,
    metric: SpatialMetric,
) -> list[SpatialHit]:
    if not isinstance(metric, SpatialMetric):
        raise TypeError("metric must be a SpatialMetric")

    k = _validate_k(k)
    center = validate_cartesian_point(center, name="center")

    if root is None or root.mbr is None:
        return []

    frontier_sequence = count()
    best_sequence = count()

    frontier: list[tuple[float, int, RTreeNode]] = []
    best: list[
        tuple[
            tuple[float, float, float, int, int],
            int,
            SpatialHit,
        ]
    ] = []

    root_lower_bound = metric.lower_bound(center, root.mbr)

    heapq.heappush(
        frontier,
        (
            root_lower_bound,
            next(frontier_sequence),
            root,
        ),
    )

    tau = math.inf

    while frontier:
        lower_bound, _sequence, node = heapq.heappop(frontier)

        if len(best) == k and lower_bound > tau:
            break

        if node.is_leaf:
            for point, rid in node.entries:
                distance = metric.distance(center, point)
                hit = SpatialHit(
                    point=point,
                    rid=rid,
                    distance=distance,
                )
                rank = _rank(hit)
                heap_item = (
                    _negative_rank(rank),
                    next(best_sequence),
                    hit,
                )

                if len(best) < k:
                    heapq.heappush(best, heap_item)
                else:
                    worst_rank = _positive_rank(best[0][0])

                    if rank < worst_rank:
                        heapq.heapreplace(best, heap_item)

            if len(best) == k:
                tau = _positive_rank(best[0][0])[0]

            continue

        for child_mbr, child in iter_child_nodes(node):
            child_lower_bound = metric.lower_bound(
                center,
                child_mbr,
            )

            if len(best) < k or child_lower_bound <= tau:
                heapq.heappush(
                    frontier,
                    (
                        child_lower_bound,
                        next(frontier_sequence),
                        child,
                    ),
                )

    hits = [item[2] for item in best]
    hits.sort(key=_rank)

    return hits


def knn_search(
    root: RTreeNode | None,
    center: Point,
    k: int,
    metric: SpatialMetric,
) -> list[RID]:
    return [
        hit.rid
        for hit in knn_hits(
            root,
            center,
            k,
            metric,
        )
    ]
