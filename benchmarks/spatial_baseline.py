"""Baseline espacial secuencial: recorre todas las entradas en cada consulta.

Sirve como oráculo de búsqueda lineal frente a R-Tree y PostgreSQL GiST, por lo
que no usa índices, poda ni ordenamiento previo. La métrica de distancia la
aporta el llamador.
"""

import heapq
from collections.abc import Callable, Iterable

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point

DistanceFn = Callable[[Point, Point], float]


class SequentialSpatialScan:
    def __init__(self, entries: Iterable[tuple[Point, RID]] = ()) -> None:
        self._entries: list[tuple[Point, RID]] = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def point_search(self, point: Point) -> list[RID]:
        return [rid for stored, rid in self._entries if stored == point]

    def range_search(self, box: MBR) -> list[RID]:
        return [rid for stored, rid in self._entries if box.contains(stored)]

    def radius_search(self, center: Point, radius: float, distance_fn: DistanceFn) -> list[RID]:
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        return [rid for stored, rid in self._entries if distance_fn(center, stored) <= radius]

    def knn(self, center: Point, k: int, distance_fn: DistanceFn) -> list[RID]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # Desempate determinista: (distancia, Point, RID) se comparan como tuplas,
        # es decir por x, y, page_id y slot, independientemente del orden de inserción.
        scored = [(distance_fn(center, stored), stored, rid) for stored, rid in self._entries]
        return [rid for _distance, _point, rid in heapq.nsmallest(k, scored)]
