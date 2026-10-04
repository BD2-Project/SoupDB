"""Comparación experimental 2.2.4: búsqueda secuencial contra R-Tree.

Mide las mismas consultas sobre los mismos datos con las dos técnicas, usando el
workload reproducible de ``benchmarks.spatial_workload``.

Sobre el conteo de accesos a disco
----------------------------------
Las consultas del R-Tree corren sobre el árbol en memoria: ``RTreePageStore`` hoy
persiste y carga el árbol entero, no página por página, así que un ``DiskManager``
no vería ninguna lectura durante una búsqueda. Reportar cero sería engañoso, porque
la poda por MBR es justamente lo que el experimento quiere mostrar.

Como un nodo del R-Tree ocupa exactamente una página en ``_page_store``, se cuenta
una lectura por nodo visitado. Para la búsqueda secuencial, que no tiene índice y
recorre todas las entradas, se cuentan todas las páginas de su archivo. Las dos
cifras miden lo mismo —páginas que habría que traer de disco— y por eso son
comparables entre sí.

La construcción sí se mide con un ``DiskManager`` real: ahí las escrituras y el
tamaño en disco son medidas directas, no un modelo.
"""

import statistics
import tempfile
import time
from pathlib import Path

from benchmarks.harness import BenchmarkResult, write_results
from benchmarks.spatial_baseline import SequentialSpatialScan
from benchmarks.spatial_workload import (
    BENCHMARK_K_VALUES,
    BENCHMARK_RADIUS_VALUES_M,
    build_rtree,
    build_spatial_workload,
)
from engine.algorithms.spatial import haversine_metric
from engine.algorithms.spatial.stats import SpatialQueryStats
from engine.indexes.rtree import SpatialQueries
from engine.indexes.rtree._page_store import RTreePageStore
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

# No hay constants.py en engine/common: el resto de los scripts de benchmark también
# lo declara acá.
PAGE_SIZE = 4096

# Un nodo por página: con 4096 bytes caben 127 entradas de hoja y 102 de nodo interno,
# así que 102 es el orden más grande que no deja media página vacía. Con el orden por
# defecto (8) cada nodo ocuparía una página entera para ocho entradas y la comparación
# de accesos a disco mediría ese desperdicio, no la poda del índice.
RTREE_ORDER = 102

DATASET_SIZES = (1_000, 10_000, 100_000)
RUNS = 5
QUERIES_PER_PARAMETER = 20
BUFFER_POOL_PAGES = 256
SEED = 20260404

# Una entrada persistida del scan secuencial: dos coordenadas y un RID.
SEQUENTIAL_ENTRY_BYTES = 8 + 8 + 4 + 4

SEQUENTIAL = "sequential_scan"
RTREE = "rtree"


def _median_result(
    structure: str,
    operation: str,
    records: int,
    samples: list[tuple[float, int, int, int]],
    size_bytes: int,
) -> BenchmarkResult:
    """Reduce las corridas a su mediana.

    Se usa la mediana y no el promedio porque la primera corrida suele venir
    contaminada por el cache del sistema operativo.
    """
    elapsed = statistics.median(sample[0] for sample in samples)
    reads = int(statistics.median(sample[1] for sample in samples))
    writes = int(statistics.median(sample[2] for sample in samples))
    count = int(statistics.median(sample[3] for sample in samples))
    operations = len(samples) and QUERIES_PER_PARAMETER

    return BenchmarkResult(
        structure=structure,
        operation=operation,
        records=records,
        operations=operations,
        elapsed_ms=elapsed,
        ops_per_second=(operations / (elapsed / 1000)) if elapsed > 0 else 0.0,
        disk_reads=reads,
        disk_writes=writes,
        size_bytes=size_bytes,
        result_count=count,
    )


def _sequential_pages(records: int) -> int:
    """Páginas que ocuparía el archivo plano de entradas del scan secuencial."""
    per_page = PAGE_SIZE // SEQUENTIAL_ENTRY_BYTES
    return -(-records // per_page)


def _persist_rtree(tree, directory: Path) -> tuple[int, int, int]:
    """Guarda el árbol con un DiskManager real y devuelve (lecturas, escrituras, bytes)."""
    path = directory / "rtree.idx"
    disk = DiskManager(path, page_size=PAGE_SIZE)
    buffers = BufferManager(disk, capacity=BUFFER_POOL_PAGES)
    try:
        RTreePageStore(disk, buffers).save_tree(tree)
        buffers.flush_all()
        return disk.reads, disk.writes, path.stat().st_size
    finally:
        disk.close()


def _build_phase(records: int, directory: Path) -> tuple[list[BenchmarkResult], object, object]:
    workload = build_spatial_workload(
        entry_count=records,
        query_count=QUERIES_PER_PARAMETER,
        seed=SEED,
    )

    rtree_samples: list[tuple[float, int, int, int]] = []
    tree = None
    rtree_bytes = 0
    for _ in range(RUNS):
        started = time.perf_counter()
        tree = build_rtree(workload.entries, order=RTREE_ORDER)
        elapsed = (time.perf_counter() - started) * 1000
        with tempfile.TemporaryDirectory(dir=directory) as scratch:
            reads, writes, rtree_bytes = _persist_rtree(tree, Path(scratch))
        rtree_samples.append((elapsed, reads, writes, records))

    scan_samples: list[tuple[float, int, int, int]] = []
    scan = None
    for _ in range(RUNS):
        started = time.perf_counter()
        scan = SequentialSpatialScan(workload.entries)
        elapsed = (time.perf_counter() - started) * 1000
        pages = _sequential_pages(records)
        scan_samples.append((elapsed, 0, pages, records))

    results = [
        _median_result(RTREE, "build", records, rtree_samples, rtree_bytes),
        _median_result(
            SEQUENTIAL,
            "build",
            records,
            scan_samples,
            _sequential_pages(records) * PAGE_SIZE,
        ),
    ]
    return results, workload, (tree, scan)


def _radius_phase(records, workload, tree, scan, rtree_bytes, scan_bytes) -> list[BenchmarkResult]:
    metric = haversine_metric()
    queries = SpatialQueries(tree)
    stats = SpatialQueryStats()
    results = []

    for radius in BENCHMARK_RADIUS_VALUES_M:
        centers = [query.center for query in workload.radius_queries][:QUERIES_PER_PARAMETER]
        operation = f"radius_{int(radius)}m"

        samples = []
        for _ in range(RUNS):
            visited = 0
            found = 0
            started = time.perf_counter()
            for center in centers:
                hits = queries.radius_search(center, radius, metric, stats=stats)
                visited += stats.nodes_visited
                found += len(hits)
            elapsed = (time.perf_counter() - started) * 1000
            samples.append((elapsed, visited, 0, found))
        results.append(_median_result(RTREE, operation, records, samples, rtree_bytes))

        samples = []
        pages = _sequential_pages(records)
        for _ in range(RUNS):
            found = 0
            started = time.perf_counter()
            for center in centers:
                found += len(scan.radius_search(center, radius, metric.distance))
            elapsed = (time.perf_counter() - started) * 1000
            samples.append((elapsed, pages * len(centers), 0, found))
        results.append(_median_result(SEQUENTIAL, operation, records, samples, scan_bytes))

    return results


def _knn_phase(records, workload, tree, scan, rtree_bytes, scan_bytes) -> list[BenchmarkResult]:
    metric = haversine_metric()
    queries = SpatialQueries(tree)
    stats = SpatialQueryStats()
    results = []

    for k in BENCHMARK_K_VALUES:
        centers = [query.center for query in workload.knn_queries][:QUERIES_PER_PARAMETER]
        operation = f"knn_{k}"

        samples = []
        for _ in range(RUNS):
            visited = 0
            found = 0
            started = time.perf_counter()
            for center in centers:
                hits = queries.knn(center, k, metric, stats=stats)
                visited += stats.nodes_visited
                found += len(hits)
            elapsed = (time.perf_counter() - started) * 1000
            samples.append((elapsed, visited, 0, found))
        results.append(_median_result(RTREE, operation, records, samples, rtree_bytes))

        samples = []
        pages = _sequential_pages(records)
        for _ in range(RUNS):
            found = 0
            started = time.perf_counter()
            for center in centers:
                found += len(scan.knn(center, k, metric.distance))
            elapsed = (time.perf_counter() - started) * 1000
            samples.append((elapsed, pages * len(centers), 0, found))
        results.append(_median_result(SEQUENTIAL, operation, records, samples, scan_bytes))

    return results


def run(sizes: tuple[int, ...] = DATASET_SIZES) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as workspace:
        directory = Path(workspace)
        for records in sizes:
            build, workload, (tree, scan) = _build_phase(records, directory)
            results.extend(build)

            rtree_bytes = build[0].size_bytes
            scan_bytes = build[1].size_bytes
            results.extend(_radius_phase(records, workload, tree, scan, rtree_bytes, scan_bytes))
            results.extend(_knn_phase(records, workload, tree, scan, rtree_bytes, scan_bytes))

    return results


def main() -> None:
    results = run()
    path = write_results("spatial_indexes", results)
    print(f"{len(results)} mediciones escritas en {path}")


if __name__ == "__main__":
    main()
