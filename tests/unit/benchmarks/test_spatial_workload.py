from benchmarks.spatial_workload import (
    BENCHMARK_K_VALUES,
    BENCHMARK_RADIUS_VALUES_M,
    KNNWorkloadQuery,
    RadiusWorkloadQuery,
    SpatialWorkload,
    build_rtree,
    build_spatial_workload,
    generate_geographic_entries,
    validate_spatial_workload,
    workload_is_correct,
)
from engine.common.rid import RID
from engine.indexes.rtree import Point, RTree


def test_workload_is_reproducible_for_same_seed() -> None:
    first = build_spatial_workload(
        entry_count=50,
        query_count=9,
        seed=42,
    )
    second = build_spatial_workload(
        entry_count=50,
        query_count=9,
        seed=42,
    )

    assert first == second


def test_different_seed_changes_generated_entries() -> None:
    first = generate_geographic_entries(
        20,
        seed=1,
    )
    second = generate_geographic_entries(
        20,
        seed=2,
    )

    assert first != second


def test_query_centers_do_not_depend_on_dataset_size() -> None:
    small = build_spatial_workload(
        entry_count=20,
        query_count=6,
        seed=123,
    )
    large = build_spatial_workload(
        entry_count=200,
        query_count=6,
        seed=123,
    )

    assert tuple(query.center for query in small.radius_queries) == tuple(
        query.center for query in large.radius_queries
    )

    assert tuple(query.center for query in small.knn_queries) == tuple(
        query.center for query in large.knn_queries
    )


def test_radius_queries_cycle_required_values() -> None:
    workload = build_spatial_workload(
        entry_count=20,
        query_count=6,
        seed=7,
    )

    assert tuple(query.radius_m for query in workload.radius_queries) == (
        BENCHMARK_RADIUS_VALUES_M * 2
    )


def test_knn_queries_cycle_required_values() -> None:
    workload = build_spatial_workload(
        entry_count=20,
        query_count=6,
        seed=7,
    )

    assert tuple(query.k for query in workload.knn_queries) == (BENCHMARK_K_VALUES * 2)


def test_generated_rids_are_unique() -> None:
    entries = generate_geographic_entries(
        300,
        seed=99,
    )

    rids = [rid for _point, rid in entries]

    assert len(rids) == len(set(rids))


def test_workload_matches_sequential_baseline() -> None:
    workload = build_spatial_workload(
        entry_count=100,
        query_count=9,
        seed=2026,
    )
    tree = build_rtree(
        workload.entries,
        order=4,
    )

    results = validate_spatial_workload(
        workload,
        tree,
    )

    assert len(results) == 18
    assert workload_is_correct(results)


def test_validation_detects_incomplete_tree() -> None:
    point = Point(-77.0428, -12.0464)

    workload = SpatialWorkload(
        seed=1,
        entries=(
            (point, RID(0, 0)),
            (Point(-77.0430, -12.0465), RID(0, 1)),
        ),
        radius_queries=(
            RadiusWorkloadQuery(
                query_id=0,
                center=point,
                radius_m=10_000.0,
            ),
        ),
        knn_queries=(
            KNNWorkloadQuery(
                query_id=0,
                center=point,
                k=2,
            ),
        ),
    )

    incomplete_tree = RTree(order=2)
    incomplete_tree.insert(
        point,
        RID(0, 0),
    )

    results = validate_spatial_workload(
        workload,
        incomplete_tree,
    )

    assert not workload_is_correct(results)
    assert any(not result.correct for result in results)


def test_validation_preserves_duplicate_entries() -> None:
    point = Point(-77.0428, -12.0464)

    entries = (
        (point, RID(0, 0)),
        (point, RID(0, 0)),
        (point, RID(0, 1)),
    )

    workload = SpatialWorkload(
        seed=1,
        entries=entries,
        radius_queries=(
            RadiusWorkloadQuery(
                query_id=0,
                center=point,
                radius_m=0.0,
            ),
        ),
        knn_queries=(
            KNNWorkloadQuery(
                query_id=0,
                center=point,
                k=3,
            ),
        ),
    )

    tree = build_rtree(
        workload.entries,
        order=2,
    )

    results = validate_spatial_workload(
        workload,
        tree,
    )

    assert workload_is_correct(results)

    radius_result = next(result for result in results if result.query_type == "radius")
    knn_result = next(result for result in results if result.query_type == "knn")

    assert radius_result.expected_count == 3
    assert radius_result.actual_count == 3
    assert knn_result.expected_count == 3
    assert knn_result.actual_count == 3
