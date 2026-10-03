from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from benchmarks.spatial_baseline import SequentialSpatialScan
from engine.algorithms.spatial import SpatialMetric, haversine_metric
from engine.common.rid import RID
from engine.indexes.rtree import Point, RTree, SpatialQueries

BENCHMARK_DATASET_SIZES = (1_000, 10_000, 100_000)
BENCHMARK_RADIUS_VALUES_M = (1_000.0, 5_000.0, 10_000.0)
BENCHMARK_K_VALUES = (10, 50, 100)
BENCHMARK_QUERY_COUNT = 100

DEFAULT_LON_MIN = -77.20
DEFAULT_LON_MAX = -76.90
DEFAULT_LAT_MIN = -12.20
DEFAULT_LAT_MAX = -11.90

_QUERY_SEED_OFFSET = 1_000_003


@dataclass(frozen=True)
class RadiusWorkloadQuery:
    query_id: int
    center: Point
    radius_m: float


@dataclass(frozen=True)
class KNNWorkloadQuery:
    query_id: int
    center: Point
    k: int


@dataclass(frozen=True)
class SpatialWorkload:
    seed: int
    entries: tuple[tuple[Point, RID], ...]
    radius_queries: tuple[RadiusWorkloadQuery, ...]
    knn_queries: tuple[KNNWorkloadQuery, ...]


@dataclass(frozen=True)
class SpatialValidationResult:
    query_id: int
    query_type: str
    parameter: float | int
    correct: bool
    expected_count: int
    actual_count: int


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")

    if value <= 0:
        raise ValueError(f"{name} must be positive")

    return value


def generate_geographic_entries(
    count: int,
    *,
    seed: int,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
) -> tuple[tuple[Point, RID], ...]:
    count = _validate_positive_int(count, name="count")

    if lon_min >= lon_max:
        raise ValueError("lon_min must be less than lon_max")

    if lat_min >= lat_max:
        raise ValueError("lat_min must be less than lat_max")

    rng = random.Random(seed)
    entries = []

    for index in range(count):
        point = Point(
            rng.uniform(lon_min, lon_max),
            rng.uniform(lat_min, lat_max),
        )
        rid = RID(
            index // 128,
            index % 128,
        )
        entries.append((point, rid))

    return tuple(entries)


def generate_query_centers(
    count: int,
    *,
    seed: int,
    lon_min: float = DEFAULT_LON_MIN,
    lon_max: float = DEFAULT_LON_MAX,
    lat_min: float = DEFAULT_LAT_MIN,
    lat_max: float = DEFAULT_LAT_MAX,
) -> tuple[Point, ...]:
    count = _validate_positive_int(count, name="count")

    rng = random.Random(seed + _QUERY_SEED_OFFSET)

    return tuple(
        Point(
            rng.uniform(lon_min, lon_max),
            rng.uniform(lat_min, lat_max),
        )
        for _ in range(count)
    )


def build_spatial_workload(
    *,
    entry_count: int,
    query_count: int,
    seed: int,
) -> SpatialWorkload:
    entries = generate_geographic_entries(
        entry_count,
        seed=seed,
    )
    centers = generate_query_centers(
        query_count,
        seed=seed,
    )

    radius_queries = tuple(
        RadiusWorkloadQuery(
            query_id=query_id,
            center=center,
            radius_m=BENCHMARK_RADIUS_VALUES_M[query_id % len(BENCHMARK_RADIUS_VALUES_M)],
        )
        for query_id, center in enumerate(centers)
    )

    knn_queries = tuple(
        KNNWorkloadQuery(
            query_id=query_id,
            center=center,
            k=BENCHMARK_K_VALUES[query_id % len(BENCHMARK_K_VALUES)],
        )
        for query_id, center in enumerate(centers)
    )

    return SpatialWorkload(
        seed=seed,
        entries=entries,
        radius_queries=radius_queries,
        knn_queries=knn_queries,
    )


def build_rtree(
    entries: tuple[tuple[Point, RID], ...],
    *,
    order: int = 8,
) -> RTree:
    tree = RTree(order=order)

    for point, rid in entries:
        tree.insert(point, rid)

    return tree


def validate_spatial_workload(
    workload: SpatialWorkload,
    tree: RTree,
    *,
    metric: SpatialMetric | None = None,
) -> list[SpatialValidationResult]:
    if metric is None:
        metric = haversine_metric()

    baseline = SequentialSpatialScan(workload.entries)
    indexed = SpatialQueries(tree)
    results = []

    for query in workload.radius_queries:
        expected = baseline.radius_search(
            query.center,
            query.radius_m,
            metric.distance,
        )
        actual = indexed.radius_search(
            query.center,
            query.radius_m,
            metric,
        )

        results.append(
            SpatialValidationResult(
                query_id=query.query_id,
                query_type="radius",
                parameter=query.radius_m,
                correct=Counter(actual) == Counter(expected),
                expected_count=len(expected),
                actual_count=len(actual),
            )
        )

    for query in workload.knn_queries:
        expected = baseline.knn(
            query.center,
            query.k,
            metric.distance,
        )
        actual = indexed.knn(
            query.center,
            query.k,
            metric,
        )

        results.append(
            SpatialValidationResult(
                query_id=query.query_id,
                query_type="knn",
                parameter=query.k,
                correct=actual == expected,
                expected_count=len(expected),
                actual_count=len(actual),
            )
        )

    return results


def workload_is_correct(
    results: list[SpatialValidationResult],
) -> bool:
    return all(result.correct for result in results)
