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

Comparación con y sin índice
----------------------------
:class:`SequentialSpatialScan` es el camino *sin* índice. La misma consulta por
radio servida por el R-Tree (caja envolvente + predicado exacto encima, tal
como la planifica el motor) se mide contra él en
:func:`compare_radius_with_and_without_index`, que devuelve el coste de los dos
caminos, el número de candidatos que devuelve la caja y si ambos coinciden.
"""

import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

from engine.common.rid import RID
from engine.indexes.rtree import RTree
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point
from engine.query.spatial_metrics import (
    EARTH_RADIUS_KM,
    EUCLIDEAN,
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
    "RadiusQueryComparison",
    "SequentialSpatialScan",
    "compare_radius_with_and_without_index",
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


@dataclass(frozen=True)
class RadiusQueryComparison:
    """Coste y resultado de una misma consulta por radio, con y sin índice.

    ``matches`` es el número de filas que devuelve cada camino (deben coincidir:
    ``agrees`` lo verifica). ``index_candidates`` es lo que el R-Tree devuelve
    por la caja envolvente *antes* del filtro exacto, así que mide la
    selectividad de la caja frente al círculo: es el trabajo de más que hace el
    índice y el que se ahorra cuando la caja es ajustada.
    """

    entries: int
    matches: int
    index_candidates: int
    index_ms: float
    scan_ms: float
    agrees: bool

    @property
    def speedup(self) -> float:
        """Veces que el camino indexado tardó menos (mayor que 1 = índice gana)."""
        if self.index_ms == 0.0:
            return float("inf") if self.scan_ms > 0.0 else 1.0
        return self.scan_ms / self.index_ms


def compare_radius_with_and_without_index(
    entries: Iterable[tuple[Point, RID]],
    center: Point,
    radius: float,
    *,
    metric: str = EUCLIDEAN,
    order: int = 4,
    repeats: int = 1,
) -> RadiusQueryComparison:
    """Ejecuta la consulta por radio con R-Tree y con escaneo secuencial.

    Reproduce exactamente lo que hace el planner: el camino indexado pide la
    caja envolvente ``center ± radius`` al R-Tree y luego aplica el predicado
    exacto encima (una caja no es un círculo), mientras que el baseline evalúa
    la distancia de cada entrada. Los dos conjuntos de RIDs deben coincidir, y
    la comparación lo dice en ``agrees`` en vez de asumirlo.

    Solo admite la métrica euclidiana: es la única cuyo límite es un rectángulo
    alineado a los ejes que contiene al círculo, y por tanto la única que puede
    aprovechar una búsqueda por rango sin riesgo de perder filas. Con
    ``'haversine'`` el radio está en kilómetros sobre una esfera y su equivalente
    en grados depende de la latitud de cada fila, así que el índice no puede
    acotar los candidatos y el motor mantiene el escaneo completo.
    """
    canonical = normalize_metric(metric)
    if canonical != EUCLIDEAN:
        raise ValueError(
            "the indexed radius path only supports the euclidean metric, since a "
            f"haversine radius has no latitude independent bounding box (got {metric!r})"
        )
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")

    entries = list(entries)
    distance_fn = distance_function(canonical)
    scan = SequentialSpatialScan(entries)
    tree = RTree(order=order)
    for point, rid in entries:
        tree.insert(point, rid)

    lo = (center.x - radius, center.y - radius)
    hi = (center.x + radius, center.y + radius)

    def indexed() -> list[RID]:
        by_rid = {rid: point for point, rid in entries}
        return [
            rid for rid in tree.range_search(lo, hi) if distance_fn(center, by_rid[rid]) <= radius
        ]

    indexed_ms, indexed_result = _best_ms(indexed, repeats)

    def sequential() -> list[RID]:
        return scan.radius_search(center, radius, distance_fn)

    scan_ms, scan_result = _best_ms(sequential, repeats)

    return RadiusQueryComparison(
        entries=len(entries),
        matches=len(scan_result),
        index_candidates=len(tree.range_search(lo, hi)),
        index_ms=indexed_ms,
        scan_ms=scan_ms,
        agrees=sorted(indexed_result) == sorted(scan_result),
    )


def _best_ms(operation: Callable[[], object], repeats: int) -> tuple[float, object]:
    """Milisegundos de la mejor ejecución de ``repeats`` y su resultado."""
    best = float("inf")
    result: object = None
    for _ in range(repeats):
        started = perf_counter()
        result = operation()
        best = min(best, (perf_counter() - started) * 1000)
    return best, result
