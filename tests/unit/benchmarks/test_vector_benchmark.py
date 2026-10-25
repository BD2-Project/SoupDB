"""Smoke test of the vector benchmark: tiny sizes, every column filled."""

from benchmarks.scripts.compare_vector_indexes import (
    Config,
    query_vectors,
    run,
    summarize,
    synthetic_vectors,
    time_vs_size_svg,
    write_results,
)


def test_same_seed_gives_same_vectors_and_queries() -> None:
    items = synthetic_vectors(50, 4)

    assert items == synthetic_vectors(50, 4)
    assert query_vectors(items, 5) == query_vectors(items, 5)


def test_run_measures_every_technique_and_metric(tmp_path) -> None:
    config = Config(dim=8, queries=5, runs=2, m=4, ef_construction=16, ef_search=16)

    results = run([200, 400], config=config, progress=lambda _msg: None)

    combos = {(r.technique, r.metric, r.dataset_size) for r in results}
    assert len(combos) == 3 * 2 * 2
    knn = [r for r in results if r.operation == "knn"]
    assert all(0.0 <= r.recall_at_k <= 1.0 for r in knn)
    assert all(r.recall_at_k == 1.0 for r in knn if r.technique == "brute_force")
    assert all(r.disk_reads > 0 for r in knn)
    assert all(r.bytes_on_disk > 0 for r in results)
    assert len(summarize(results)) == 3 * 2 * 2 * 2

    path = write_results(results, tmp_path)
    assert (
        path.read_text()
        .splitlines()[0]
        .startswith(
            "technique,operation,dataset_size,run,elapsed_ms,disk_reads,disk_writes,bytes_on_disk"
        )
    )
    assert time_vs_size_svg(results, "knn", "k-NN").startswith("<svg")
