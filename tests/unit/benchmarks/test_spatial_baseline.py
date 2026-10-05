import math
import random

import pytest

from benchmarks.spatial_baseline import (
    EARTH_RADIUS_M,
    EUCLIDEAN_DISTANCE,
    HAVERSINE_DISTANCE,
    METRIC_DISTANCES,
    SequentialSpatialScan,
    distance_function,
    metric_functions,
)
from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point
from engine.query.spatial_metrics import normalize_metric


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def haversine_oracle(a: Point, b: Point) -> float:
    """Independent geodesic formula (atan2 form) to cross-check the baseline."""
    phi1, phi2 = math.radians(a.y), math.radians(b.y)
    d_phi, d_lambda = phi2 - phi1, math.radians(b.x - a.x)
    h = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(h), math.sqrt(1 - h))


class CountingDistance:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, a: Point, b: Point) -> float:
        self.calls += 1
        return euclidean(a, b)


class CountingMBR(MBR):
    calls = 0

    def contains(self, point: Point) -> bool:
        type(self).calls += 1
        return super().contains(point)


def _scan(*points: tuple[float, float]) -> SequentialSpatialScan:
    return SequentialSpatialScan((Point(x, y), RID(i, 0)) for i, (x, y) in enumerate(points))


GRID = _scan((0, 0), (1, 0), (0, 1), (3, 4), (-2, -2), (10, 10))


def test_empty_collection() -> None:
    scan = SequentialSpatialScan()
    assert len(scan) == 0
    assert scan.point_search(Point(0, 0)) == []
    assert scan.range_search(MBR(-1, -1, 1, 1)) == []
    assert scan.radius_search(Point(0, 0), 5, euclidean) == []
    assert scan.knn(Point(0, 0), 3, euclidean) == []


def test_point_search_existing_and_missing() -> None:
    assert GRID.point_search(Point(3, 4)) == [RID(3, 0)]
    assert GRID.point_search(Point(3, 5)) == []


def test_point_search_duplicates_with_different_rids() -> None:
    scan = SequentialSpatialScan(
        [(Point(1, 1), RID(0, 0)), (Point(2, 2), RID(0, 1)), (Point(1, 1), RID(5, 3))]
    )
    assert scan.point_search(Point(1, 1)) == [RID(0, 0), RID(5, 3)]


def test_repeated_entries_are_kept() -> None:
    entry = (Point(1, 1), RID(0, 0))
    scan = SequentialSpatialScan([entry, entry, entry])
    assert len(scan) == 3
    assert scan.point_search(Point(1, 1)) == [RID(0, 0)] * 3
    assert scan.range_search(MBR(0, 0, 2, 2)) == [RID(0, 0)] * 3
    assert scan.radius_search(Point(0, 0), 2, euclidean) == [RID(0, 0)] * 3
    assert scan.knn(Point(0, 0), 2, euclidean) == [RID(0, 0)] * 2


def test_range_search_rectangle() -> None:
    assert GRID.range_search(MBR(-0.5, -0.5, 3.5, 4.5)) == [
        RID(0, 0),
        RID(1, 0),
        RID(2, 0),
        RID(3, 0),
    ]
    assert GRID.range_search(MBR(100, 100, 200, 200)) == []


def test_range_search_borders_are_inclusive() -> None:
    assert GRID.range_search(MBR(0, 0, 1, 1)) == [RID(0, 0), RID(1, 0), RID(2, 0)]
    assert GRID.range_search(MBR(3, 4, 3, 4)) == [RID(3, 0)]
    assert GRID.range_search(MBR(-2, -2, -2, 10)) == [RID(4, 0)]


def test_radius_search_inside_and_outside() -> None:
    assert GRID.radius_search(Point(0, 0), 1.5, euclidean) == [RID(0, 0), RID(1, 0), RID(2, 0)]
    assert GRID.radius_search(Point(50, 50), 1, euclidean) == []


def test_radius_search_includes_exact_radius() -> None:
    assert GRID.radius_search(Point(0, 0), 5, euclidean) == [
        RID(0, 0),
        RID(1, 0),
        RID(2, 0),
        RID(3, 0),
        RID(4, 0),
    ]


def test_radius_zero_matches_only_same_point() -> None:
    assert GRID.radius_search(Point(3, 4), 0, euclidean) == [RID(3, 0)]
    assert GRID.radius_search(Point(3, 3), 0, euclidean) == []


def test_negative_radius_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        GRID.radius_search(Point(0, 0), -0.1, euclidean)


def test_radius_uses_given_distance_function() -> None:
    def manhattan(a: Point, b: Point) -> float:
        return abs(a.x - b.x) + abs(a.y - b.y)

    assert GRID.radius_search(Point(0, 0), 1, manhattan) == [RID(0, 0), RID(1, 0), RID(2, 0)]
    assert GRID.radius_search(Point(0, 0), 5, manhattan) == [
        RID(0, 0),
        RID(1, 0),
        RID(2, 0),
        RID(4, 0),
    ]


def test_knn_single() -> None:
    assert GRID.knn(Point(2.9, 3.9), 1, euclidean) == [RID(3, 0)]


def test_knn_several_sorted_by_distance() -> None:
    assert GRID.knn(Point(9, 9), 3, euclidean) == [RID(5, 0), RID(3, 0), RID(2, 0)]


def test_knn_k_larger_than_collection_returns_all() -> None:
    result = GRID.knn(Point(0, 0), 100, euclidean)
    assert result == [RID(0, 0), RID(2, 0), RID(1, 0), RID(4, 0), RID(3, 0), RID(5, 0)]


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        GRID.knn(Point(0, 0), k, euclidean)


def test_knn_tie_break_is_deterministic() -> None:
    entries = [
        (Point(0, 1), RID(9, 9)),
        (Point(1, 0), RID(7, 1)),
        (Point(-1, 0), RID(3, 3)),
        (Point(0, -1), RID(4, 4)),
        (Point(1, 0), RID(7, 0)),
    ]
    expected = [RID(3, 3), RID(4, 4), RID(9, 9), RID(7, 0), RID(7, 1)]
    for seed in range(10):
        shuffled = entries[:]
        random.Random(seed).shuffle(shuffled)
        scan = SequentialSpatialScan(shuffled)
        assert scan.knn(Point(0, 0), 5, euclidean) == expected
        assert scan.knn(Point(0, 0), 2, euclidean) == expected[:2]


def test_radius_and_knn_evaluate_every_entry() -> None:
    scan = _scan(*[(i, i) for i in range(50)])

    distance = CountingDistance()
    scan.radius_search(Point(0, 0), 0, distance)
    assert distance.calls == 50

    distance = CountingDistance()
    scan.knn(Point(0, 0), 1, distance)
    assert distance.calls == 50


def test_range_evaluates_every_entry() -> None:
    CountingMBR.calls = 0
    _scan(*[(i, i) for i in range(50)]).range_search(CountingMBR(0, 0, 0, 0))
    assert CountingMBR.calls == 50


def test_constructor_copies_input() -> None:
    entries = [(Point(1, 1), RID(0, 0))]
    scan = SequentialSpatialScan(entries)
    entries.append((Point(1, 1), RID(0, 1)))
    assert scan.point_search(Point(1, 1)) == [RID(0, 0)]


def _dataset(seed: int, size: int) -> list[tuple[Point, RID]]:
    rng = random.Random(seed)
    # Coordenadas enteras en un rango pequeño para forzar duplicados y empates.
    return [
        (Point(rng.randint(-20, 20), rng.randint(-20, 20)), RID(rng.randint(0, 5), i))
        for i in range(size)
    ]


@pytest.mark.parametrize(("seed", "size"), [(1, 1), (2, 10), (3, 200), (4, 500), (5, 1000)])
def test_matches_brute_force_oracle(seed: int, size: int) -> None:
    rng = random.Random(seed * 100)
    entries = _dataset(seed, size)
    scan = SequentialSpatialScan(entries)

    for _ in range(20):
        center = Point(rng.randint(-25, 25), rng.randint(-25, 25))
        x1, x2 = sorted(rng.randint(-25, 25) for _ in range(2))
        y1, y2 = sorted(rng.randint(-25, 25) for _ in range(2))
        box = MBR(x1, y1, x2, y2)
        radius = rng.choice([0, 1, 5, 12.5, 40])
        k = rng.randint(1, size + 3)

        assert scan.point_search(center) == [rid for p, rid in entries if p == center]
        assert scan.range_search(box) == [
            rid for p, rid in entries if x1 <= p.x <= x2 and y1 <= p.y <= y2
        ]
        assert scan.radius_search(center, radius, euclidean) == [
            rid for p, rid in entries if euclidean(center, p) <= radius
        ]
        ranked = sorted(entries, key=lambda e: (euclidean(center, e[0]), e[0], e[1]))
        assert scan.knn(center, k, euclidean) == [rid for _p, rid in ranked[:k]]


# --- segunda métrica: haversine ------------------------------------------


#: Capitales como Point(lon, lat); el RID es el identificador estable del R-Tree.
CITIES = {
    "Montevideo": Point(-56.1645, -34.9011),
    "Buenos Aires": Point(-58.3816, -34.6037),
    "Santiago": Point(-70.6483, -33.4489),
    "Lisboa": Point(-9.1393, 38.7223),
    "Paris": Point(2.3522, 48.8566),
}
CITY_SCAN = SequentialSpatialScan((point, RID(i, 0)) for i, point in enumerate(CITIES.values()))
MONTEVIDEO = CITY_SCAN.knn(Point(-56.1645, -34.9011), 1, HAVERSINE_DISTANCE)[0]

#: Distancias de referencia desde Montevideo (metros, radio medio 6 371 008,8).
MVD_BUE_M = 205_232.35938356873
MVD_SCL_M = 1_340_979.3901132298
MVD_LIS_M = 9_508_412.014033962
MVD_PAR_M = 10_960_801.628543702


def test_baseline_exposes_both_metrics() -> None:
    assert set(METRIC_DISTANCES) == {"euclidean", "haversine"}
    assert METRIC_DISTANCES["euclidean"] is EUCLIDEAN_DISTANCE
    assert METRIC_DISTANCES["haversine"] is HAVERSINE_DISTANCE
    assert distance_function() is EUCLIDEAN_DISTANCE
    assert distance_function("haversine") is HAVERSINE_DISTANCE
    assert distance_function("Haversine") is HAVERSINE_DISTANCE
    assert distance_function(normalize_metric("haversine")) is HAVERSINE_DISTANCE
    assert set(metric_functions()) == {EUCLIDEAN_DISTANCE, HAVERSINE_DISTANCE}


def test_baseline_shares_the_euclidean_convention_with_the_engine() -> None:
    # El baseline importa la métrica del motor: mismo resultado bit a bit.
    for a in CITIES.values():
        for b in CITIES.values():
            assert EUCLIDEAN_DISTANCE(a, b) == euclidean(a, b)


def test_haversine_radius_in_metres() -> None:
    inside = CITY_SCAN.radius_search(Point(-56.1645, -34.9011), 300_000.0, HAVERSINE_DISTANCE)
    assert inside == [MONTEVIDEO, RID(1, 0)]
    assert CITY_SCAN.radius_search(Point(-56.1645, -34.9011), 1_000_000.0, HAVERSINE_DISTANCE) == [
        MONTEVIDEO,
        RID(1, 0),
    ]
    assert CITY_SCAN.radius_search(Point(-56.1645, -34.9011), 2_000_000.0, HAVERSINE_DISTANCE) == [
        MONTEVIDEO,
        RID(1, 0),
        RID(2, 0),
    ]
    assert CITY_SCAN.radius_search(Point(0, 60), 1_000.0, HAVERSINE_DISTANCE) == []


def test_haversine_radius_zero_matches_only_the_same_point() -> None:
    assert CITY_SCAN.radius_search(Point(-56.1645, -34.9011), 0, HAVERSINE_DISTANCE) == [MONTEVIDEO]
    assert CITY_SCAN.radius_search(Point(-56.1645, -34.9010), 0, HAVERSINE_DISTANCE) == []


def test_haversine_antipodal_pair_at_half_circumference() -> None:
    antipodal = SequentialSpatialScan(
        [(Point(0, 0), RID(0, 0)), (Point(180, 0), RID(1, 0)), (Point(-180, 0), RID(2, 0))]
    )
    half_circumference = math.pi * EARTH_RADIUS_M
    # Los tres son antipodales del origen: los tres caen dentro del radio.
    assert antipodal.radius_search(Point(0, 0), half_circumference, HAVERSINE_DISTANCE) == [
        RID(0, 0),
        RID(1, 0),
        RID(2, 0),
    ]
    assert antipodal.radius_search(Point(0, 0), half_circumference - 0.001, HAVERSINE_DISTANCE) == [
        RID(0, 0)
    ]
    assert antipodal.radius_search(Point(0, 0), half_circumference, HAVERSINE_DISTANCE) == [
        RID(0, 0),
        RID(1, 0),
        RID(2, 0),
    ]
    # El origen está a 0 km; los otros dos empatan a media circunferencia y el
    # desempate sigue siendo determinista: (Point, RID) ordena -180 antes que 180.
    assert antipodal.knn(Point(0, 0), 2, HAVERSINE_DISTANCE) == [RID(0, 0), RID(2, 0)]


def test_haversine_knn_returns_nearest_first() -> None:
    nearest = CITY_SCAN.knn(Point(-56.1645, -34.9011), 4, HAVERSINE_DISTANCE)
    assert nearest == [MONTEVIDEO, RID(1, 0), RID(2, 0), RID(3, 0)]
    assert CITY_SCAN.knn(Point(-56.1645, -34.9011), 99, HAVERSINE_DISTANCE) == [
        MONTEVIDEO,
        RID(1, 0),
        RID(2, 0),
        RID(3, 0),
        RID(4, 0),
    ]


def test_haversine_knn_tie_break_is_deterministic() -> None:
    entries = [
        (Point(0, 1), RID(9, 9)),
        (Point(1, 0), RID(7, 1)),
        (Point(-1, 0), RID(3, 3)),
        (Point(0, -1), RID(4, 4)),
        (Point(1, 0), RID(7, 0)),
    ]
    expected = [RID(3, 3), RID(4, 4), RID(9, 9), RID(7, 0), RID(7, 1)]
    for seed in range(10):
        shuffled = entries[:]
        random.Random(seed).shuffle(shuffled)
        scan = SequentialSpatialScan(shuffled)
        assert scan.knn(Point(0, 0), 5, HAVERSINE_DISTANCE) == expected
        assert scan.knn(Point(0, 0), 2, HAVERSINE_DISTANCE) == expected[:2]


def test_both_metrics_can_be_compared_on_the_same_dataset() -> None:
    """Mismo radio, dos métricas: en grados entra el vecino, en km no."""
    center = Point(-56.1645, -34.9011)
    in_degrees = CITY_SCAN.radius_search(center, 3.0, EUCLIDEAN_DISTANCE)
    in_kilometres = CITY_SCAN.radius_search(center, 3.0, HAVERSINE_DISTANCE)
    assert in_degrees == [MONTEVIDEO, RID(1, 0)]
    assert in_kilometres == [MONTEVIDEO]
    # El subconjunto en km nunca es un superconjunto ni un subconjunto del de grados.
    assert set(in_kilometres) < set(in_degrees)


def test_metric_distance_functions_evaluate_every_entry() -> None:
    scan = _scan(*[(i, i) for i in range(50)])
    for distance_fn in metric_functions():
        scan.radius_search(Point(0, 0), 1000.0, distance_fn)
        scan.knn(Point(0, 0), 1, distance_fn)


def test_haversine_matches_brute_force_oracle() -> None:
    rng = random.Random(7)
    entries = [
        (
            Point(
                round(rng.uniform(-180.0, 180.0), 4),
                round(rng.uniform(-90.0, 90.0), 4),
            ),
            RID(rng.randint(0, 5), i),
        )
        for i in range(200)
    ]
    scan = SequentialSpatialScan(entries)

    for _ in range(20):
        center = Point(round(rng.uniform(-180.0, 180.0), 4), round(rng.uniform(-90.0, 90.0), 4))
        radius = rng.choice([0, 1, 100, 1000, 5000, 10000, 20015])
        k = rng.randint(1, len(entries) + 3)

        assert scan.radius_search(center, radius, HAVERSINE_DISTANCE) == [
            rid for p, rid in entries if haversine_oracle(center, p) <= radius
        ]
        ranked = sorted(
            entries,
            key=lambda e: (round(haversine_oracle(center, e[0]), 9), e[0], e[1]),
        )
        assert scan.knn(center, k, HAVERSINE_DISTANCE) == [rid for _p, rid in ranked[:k]]


def test_haversine_rejects_out_of_range_coordinates() -> None:
    scan = SequentialSpatialScan([(Point(0, 0), RID(0, 0))])
    with pytest.raises(ValueError, match=r"longitude in \[-180, 180\]"):
        scan.radius_search(Point(200, 0), 10.0, HAVERSINE_DISTANCE)
    with pytest.raises(ValueError, match=r"latitude in \[-90, 90\]"):
        scan.knn(Point(0, 91), 1, HAVERSINE_DISTANCE)
    # El radio sigue validándose antes de calcular distancias.
    with pytest.raises(ValueError, match="radius"):
        scan.radius_search(Point(200, 0), -1.0, HAVERSINE_DISTANCE)
