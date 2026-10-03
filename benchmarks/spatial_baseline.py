"""Baseline espacial secuencial: recorre todas las entradas en cada consulta.

Sirve como oráculo de búsqueda lineal frente a R-Tree y PostgreSQL GiST, por lo
que no usa índices, poda ni ordenamiento previo. La métrica de distancia la
aporta el llamador.

Métricas
--------
``radius_search`` y ``knn`` reciben un ``distance_fn`` con la misma firma para
las dos métricas soportadas, así que el mismo baseline compara el coste con y
sin índice **y** entre métrica euclidiana y haversine:

* :data:`EUCLIDEAN_DISTANCE` (nomeclatura de SQL ``distance(a, b)``): distancia
  en el plano, en las unidades de la coordenada.
* :data:`HAVERSINE_DISTANCE` (SQL ``distance(a, b, 'haversine')``): distancia
  geodésica en kilómetros.
* :func:`distance_function` resuelve un nombre de métrica, y
  :func:`metric_functions` las expone todas.

Las funciones se importan de :mod:`engine.query.spatial_metrics` a propósito: el
oráculo y el motor deben aplicar la misma convención (lon en x, lat en y, radio
terrestre :data:`~engine.query.spatial_metrics.EARTH_RADIUS_KM`) para que una
diferencia de resultados entre R-Tree y baseline sea un fallo del índice y no un
desacuerdo de fórmula. Como el módulo es puro, importarlo no arrastra el motor.
El radio de ``radius_search`` y la distancia de ``knn`` están en la unidad de la
métrica elegida (grados para la euclidiana, km para la haversine).
"""

import heapq
from collections.abc import Callable, Iterable

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point
from engine.query.spatial_metrics import (
    EARTH_RADIUS_KM,
    euclidean_distance,
    haversine_distance,
    normalize_metric,
)

DistanceFn = Callable[[Point, Point], float]

#: ``distance(a, b)``: plano, en las unidades de la coordenada.
EUCLIDEAN_DISTANCE: DistanceFn = euclidean_distance

#: ``distance(a, b, 'haversine')``: geodésica, en kilómetros.
HAVERSINE_DISTANCE: DistanceFn = haversine_distance

#: Todas las métricas usables como ``distance_fn``, por nombre canónico.
METRIC_DISTANCES: dict[str, DistanceFn] = {
    "euclidean": EUCLIDEAN_DISTANCE,
    "haversine": HAVERSINE_DISTANCE,
}


def distance_function(metric: str = "euclidean") -> DistanceFn:
    """Devuelve la función de distancia del nombre de métrica dado.

    Acepta el mismo vocabulario que SQL (sin distinguir mayúsculas) para poder
    iterar sobre métricas desde un script de benchmark sin duplicar el mapa.
    """
    return METRIC_DISTANCES[normalize_metric(metric)]


def metric_functions() -> tuple[DistanceFn, ...]:
    """Todas las métricas soportadas, para barrerlas en un benchmark."""
    return tuple(METRIC_DISTANCES.values())


__all__ = [
    "EARTH_RADIUS_KM",
    "EUCLIDEAN_DISTANCE",
    "HAVERSINE_DISTANCE",
    "METRIC_DISTANCES",
    "DistanceFn",
    "SequentialSpatialScan",
    "distance_function",
    "metric_functions",
]


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
