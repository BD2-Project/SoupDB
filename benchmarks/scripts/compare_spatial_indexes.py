"""Comparación experimental espacial: búsqueda secuencial contra R-Tree.

Sigue el protocolo declarado en el informe (§6.3): 100 consultas por combinación,
media/mediana/p95 por consulta, las dos métricas de distancia, memoria pico y
validación de resultados contra el baseline antes de aceptar cada medición.

Sobre el conteo de accesos a disco
----------------------------------
Las consultas del R-Tree corren sobre el árbol en memoria: ``RTreePageStore``
persiste y carga el árbol entero, no página por página, así que un
``DiskManager`` no vería ninguna lectura durante una búsqueda. Reportar cero
sería engañoso, porque la poda por MBR es justamente lo que se quiere mostrar.

Como un nodo del R-Tree ocupa exactamente una página en ``_page_store``, se
cuenta una lectura por nodo visitado. Para la búsqueda secuencial, que no tiene
índice y recorre todas las entradas, se cuentan todas las páginas de su archivo.
Las dos cifras miden lo mismo —páginas que habría que traer de disco— y por eso
son comparables entre sí. La construcción sí se mide con un ``DiskManager`` real.
"""

import resource
import sys
import tempfile
import time
from pathlib import Path

from benchmarks.spatial_baseline import SequentialSpatialScan
from benchmarks.spatial_report import SpatialResult, summarize, write_spatial_results
from benchmarks.spatial_workload import (
    BENCHMARK_K_VALUES,
    BENCHMARK_RADIUS_VALUES_M,
    build_rtree,
    generate_geographic_entries,
    generate_query_centers,
)
from engine.algorithms.spatial import euclidean_metric, haversine_metric
from engine.algorithms.spatial.stats import SpatialQueryStats
from engine.indexes.rtree import SpatialQueries
from engine.indexes.rtree._page_store import RTreePageStore
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 4096
BUFFER_POOL_PAGES = 256

DATASET_SIZES = (1_000, 10_000, 100_000)
QUERIES = 100
SEED = 20260404

# Un nodo por página: con 4096 bytes caben 127 entradas de hoja y 102 de nodo
# interno. Con el orden por defecto (8) cada nodo gastaría una página entera para
# ocho entradas y el conteo de accesos mediría ese desperdicio, no la poda.
RTREE_ORDER = 102

# Una entrada persistida del scan secuencial: dos coordenadas y un RID.
SEQUENTIAL_ENTRY_BYTES = 8 + 8 + 4 + 4

# Radios equivalentes en grados para la métrica euclidiana, que opera sobre
# coordenadas y no sobre metros: un grado de latitud son ~111.32 km.
DEGREES_PER_METER = 1 / 111_320.0
BENCHMARK_RADIUS_VALUES_DEG = tuple(
    radius * DEGREES_PER_METER for radius in BENCHMARK_RADIUS_VALUES_M
)

SEQUENTIAL = "sequential_scan"
RTREE = "rtree"


def _peak_rss_bytes() -> int:
    """macOS reporta ru_maxrss en bytes y Linux en kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _sequential_pages(records: int) -> int:
    """Páginas que ocupa el archivo plano de entradas del scan secuencial."""
    per_page = PAGE_SIZE // SEQUENTIAL_ENTRY_BYTES
    return -(-records // per_page)


def _persist_rtree(tree, directory: Path) -> tuple[int, int, int]:
    """Guarda el árbol con un DiskManager real: (lecturas, escrituras, bytes)."""
    path = directory / "rtree.idx"
    disk = DiskManager(path, page_size=PAGE_SIZE)
    buffers = BufferManager(disk, capacity=BUFFER_POOL_PAGES)
    try:
        RTreePageStore(disk, buffers).save_tree(tree)
        buffers.flush_all()
        return disk.reads, disk.writes, path.stat().st_size
    finally:
        disk.close()


class _Mismatch(RuntimeError):
    """El R-Tree devolvió algo distinto del oráculo secuencial."""


def _run_combination(
    *,
    structure: str,
    operation: str,
    metric_name: str,
    parameter: float,
    records: int,
    build_ms: float,
    size_bytes: int,
    disk_writes: int,
    samples: list[float],
    reads: int,
    found: int,
) -> SpatialResult:
    mean, median, p95 = summarize(samples)
    return SpatialResult(
        structure=structure,
        operation=operation,
        metric=metric_name,
        parameter=parameter,
        records=records,
        queries=len(samples),
        build_ms=build_ms,
        elapsed_ms=median,
        mean_ms=mean,
        median_ms=median,
        p95_ms=p95,
        disk_reads=reads,
        disk_writes=disk_writes,
        size_bytes=size_bytes,
        peak_rss_bytes=_peak_rss_bytes(),
        result_count=found,
    )


def run(sizes: tuple[int, ...] = DATASET_SIZES) -> list[SpatialResult]:
    results: list[SpatialResult] = []

    with tempfile.TemporaryDirectory() as workspace:
        for records in sizes:
            entries = generate_geographic_entries(records, seed=SEED)
            centers = generate_query_centers(QUERIES, seed=SEED)

            started = time.perf_counter()
            tree = build_rtree(entries, order=RTREE_ORDER)
            rtree_build_ms = (time.perf_counter() - started) * 1000
            _, rtree_writes, rtree_bytes = _persist_rtree(tree, Path(workspace))

            started = time.perf_counter()
            scan = SequentialSpatialScan(entries)
            scan_build_ms = (time.perf_counter() - started) * 1000
            scan_pages = _sequential_pages(records)
            scan_bytes = scan_pages * PAGE_SIZE

            queries = SpatialQueries(tree)
            stats = SpatialQueryStats()

            for metric_name, metric, radii in (
                ("haversine", haversine_metric(), BENCHMARK_RADIUS_VALUES_M),
                ("euclidean", euclidean_metric(), BENCHMARK_RADIUS_VALUES_DEG),
            ):
                for label, radius in zip(BENCHMARK_RADIUS_VALUES_M, radii, strict=True):
                    operation = f"radius_{int(label)}m"
                    tree_samples, scan_samples = [], []
                    visited = tree_found = scan_found = 0

                    for center in centers:
                        started = time.perf_counter()
                        hits = queries.radius_search(center, radius, metric, stats=stats)
                        tree_samples.append((time.perf_counter() - started) * 1000)
                        visited += stats.nodes_visited
                        tree_found += len(hits)

                        started = time.perf_counter()
                        expected = scan.radius_search(center, radius, metric.distance)
                        scan_samples.append((time.perf_counter() - started) * 1000)
                        scan_found += len(expected)

                        # El baseline es el oráculo: si el índice poda de más, los
                        # tiempos no significan nada y la corrida se detiene.
                        if tree_found != scan_found:
                            raise _Mismatch(
                                f"{operation}/{metric_name} con {records} registros: "
                                f"r-tree {tree_found} vs secuencial {scan_found}"
                            )

                    results.append(
                        _run_combination(
                            structure=RTREE,
                            operation=operation,
                            metric_name=metric_name,
                            parameter=radius,
                            records=records,
                            build_ms=rtree_build_ms,
                            size_bytes=rtree_bytes,
                            disk_writes=rtree_writes,
                            samples=tree_samples,
                            reads=visited,
                            found=tree_found,
                        )
                    )
                    results.append(
                        _run_combination(
                            structure=SEQUENTIAL,
                            operation=operation,
                            metric_name=metric_name,
                            parameter=radius,
                            records=records,
                            build_ms=scan_build_ms,
                            size_bytes=scan_bytes,
                            disk_writes=scan_pages,
                            samples=scan_samples,
                            reads=scan_pages * QUERIES,
                            found=scan_found,
                        )
                    )

                for k in BENCHMARK_K_VALUES:
                    operation = f"knn_{k}"
                    tree_samples, scan_samples = [], []
                    visited = tree_found = scan_found = 0

                    for center in centers:
                        started = time.perf_counter()
                        hits = queries.knn(center, k, metric, stats=stats)
                        tree_samples.append((time.perf_counter() - started) * 1000)
                        visited += stats.nodes_visited
                        tree_found += len(hits)

                        started = time.perf_counter()
                        expected = scan.knn(center, k, metric.distance)
                        scan_samples.append((time.perf_counter() - started) * 1000)
                        scan_found += len(expected)

                        if tree_found != scan_found:
                            raise _Mismatch(
                                f"{operation}/{metric_name} con {records} registros: "
                                f"r-tree {tree_found} vs secuencial {scan_found}"
                            )

                    results.append(
                        _run_combination(
                            structure=RTREE,
                            operation=operation,
                            metric_name=metric_name,
                            parameter=k,
                            records=records,
                            build_ms=rtree_build_ms,
                            size_bytes=rtree_bytes,
                            disk_writes=rtree_writes,
                            samples=tree_samples,
                            reads=visited,
                            found=tree_found,
                        )
                    )
                    results.append(
                        _run_combination(
                            structure=SEQUENTIAL,
                            operation=operation,
                            metric_name=metric_name,
                            parameter=k,
                            records=records,
                            build_ms=scan_build_ms,
                            size_bytes=scan_bytes,
                            disk_writes=scan_pages,
                            samples=scan_samples,
                            reads=scan_pages * QUERIES,
                            found=scan_found,
                        )
                    )

    return results


def main() -> None:
    results = run()
    path = write_spatial_results("spatial_indexes", results)
    print(f"{len(results)} mediciones escritas en {path}")


if __name__ == "__main__":
    main()
