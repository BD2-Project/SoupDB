"""Tests for the with/without-index radius comparison of the spatial baseline."""

import random

import pytest

from benchmarks.spatial_baseline import (
    EUCLIDEAN_DISTANCE,
    RadiusQueryComparison,
    SequentialSpatialScan,
    compare_radius_with_and_without_index,
)
from engine.common.rid import RID
from engine.indexes.rtree.point import Point

CENTER = Point(0.0, 0.0)
SET = [
    (Point(0.0, 0.0), RID(0, 0)),
    (Point(3.0, 4.0), RID(0, 1)),
    (Point(-1.0, 0.0), RID(0, 2)),
    (Point(6.0, 8.0), RID(0, 3)),
]


def test_index_and_scan_agree() -> None:
    result = compare_radius_with_and_without_index(SET, CENTER, 4.0)
    assert isinstance(result, RadiusQueryComparison)
    assert result.entries == 4
    assert result.matches == 2
    assert result.agrees is True


def test_box_candidates_are_a_superset_of_the_circle() -> None:
    result = compare_radius_with_and_without_index(SET, CENTER, 4.0)
    # A (0,0), B (3,4) y D (-1,0) caen en la caja [-4,4]²; C (6,8) no.
    assert result.index_candidates == 3
    assert result.index_candidates > result.matches


def test_radius_zero_only_matches_the_same_point() -> None:
    result = compare_radius_with_and_without_index(SET, Point(0.0, 0.0), 0.0)
    assert result.matches == 1
    assert result.agrees is True


def test_empty_dataset() -> None:
    result = compare_radius_with_and_without_index([], CENTER, 10.0)
    assert result.entries == 0
    assert result.matches == 0
    assert result.index_candidates == 0
    assert result.agrees is True


def test_haversine_is_rejected_with_a_clear_message() -> None:
    with pytest.raises(ValueError, match="euclidean"):
        compare_radius_with_and_without_index(SET, CENTER, 10.0, metric="haversine")


def test_negative_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        compare_radius_with_and_without_index(SET, CENTER, -0.5)


def test_non_positive_repeats_is_rejected() -> None:
    with pytest.raises(ValueError, match="repeats"):
        compare_radius_with_and_without_index(SET, CENTER, 1.0, repeats=0)


def test_results_match_the_sequential_oracle_on_random_data() -> None:
    rng = random.Random(11)
    entries = [
        (Point(round(rng.uniform(-30, 30), 2), round(rng.uniform(-30, 30), 2)), RID(0, i))
        for i in range(150)
    ]
    scan = SequentialSpatialScan(entries)
    for _ in range(10):
        center = Point(round(rng.uniform(-30, 30), 2), round(rng.uniform(-30, 30), 2))
        radius = round(rng.uniform(0.0, 20.0), 2)
        result = compare_radius_with_and_without_index(entries, center, radius, repeats=2)
        expected = scan.radius_search(center, radius, EUCLIDEAN_DISTANCE)
        assert result.agrees is True
        assert result.matches == len(expected)
        assert result.index_candidates >= result.matches
        assert result.speedup >= 0.0
