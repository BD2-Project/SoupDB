import math
import random

import pytest

from benchmarks.spatial_baseline import SequentialSpatialScan
from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


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
