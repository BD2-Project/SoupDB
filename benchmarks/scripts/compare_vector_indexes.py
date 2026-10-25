"""Comparación experimental 2.4.4: IVF contra HNSW, euclidiana contra coseno.

Las tres técnicas (fuerza bruta, IVF y HNSW) se miden con los mismos vectores
y las mismas consultas, con las dos métricas. La fuerza bruta es además el
oráculo del Recall@k.

Qué se mide
-----------
- ``build``: una construcción por combinación. ``elapsed_ms`` es el tiempo
  total, ``disk_writes`` las páginas escritas, ``bytes_on_disk`` el tamaño de
  los archivos y ``memory_bytes`` la memoria que queda residente al reabrir
  el índice (los centroides del IVF; HNSW y fuerza bruta no retienen nada).
- ``knn``: ``runs`` corridas de todas las consultas. Cada corrida reabre el
  índice con el buffer pool vacío, así los ``disk_reads`` no dependen de la
  corrida anterior. ``elapsed_ms`` y ``disk_reads`` son por consulta;
  ``memory_bytes`` es el pico de memoria de Python durante las consultas,
  medido en una pasada aparte para no contaminar el tiempo.

Datos
-----
Hasta que esté ``vectores.bin`` (#108) con los descriptores reales (#120) se
usan vectores sintéticos con semilla: gaussianas alrededor de centros al azar,
que se parecen más a histogramas de descriptores que el ruido uniforme.

Uso::

    uv run python -m benchmarks.scripts.compare_vector_indexes
    uv run python -m benchmarks.scripts.compare_vector_indexes --sizes 1000 10000 100000
"""

import argparse
import csv
import math
import random
import shutil
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.vector.brute_force import BruteForceIndex
from engine.indexes.vector.constants import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M
from engine.indexes.vector.hnsw import HNSWIndex
from engine.indexes.vector.ivf import IVFIndex
from engine.indexes.vector.metrics import COSINE, EUCLIDEAN
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 4096
BUFFER_CAPACITY = 256
SEED = 20261024

DATASET_SIZES = (1_000, 10_000)
METRICS = (EUCLIDEAN, COSINE)
DIM = 64
QUERIES = 50
K = 10
RUNS = 5
CLUSTERS = 50
SPREAD = 0.5

BRUTE_FORCE = "brute_force"
IVF = "ivf"
HNSW = "hnsw"
TECHNIQUES = (BRUTE_FORCE, IVF, HNSW)

Item = tuple[list[float], RID]


@dataclass
class VectorResult:
    technique: str
    operation: str
    dataset_size: int
    run: int
    elapsed_ms: float
    disk_reads: float
    disk_writes: int
    bytes_on_disk: int
    metric: str
    dim: int
    queries: int
    k: int
    recall_at_k: float
    memory_bytes: int
    params: str


@dataclass
class Config:
    dim: int = DIM
    queries: int = QUERIES
    k: int = K
    runs: int = RUNS
    nprobe: int = 8
    m: int = HNSW_M
    ef_construction: int = HNSW_EF_CONSTRUCTION
    ef_search: int = HNSW_EF_SEARCH
    buffer_capacity: int = BUFFER_CAPACITY


def synthetic_vectors(n: int, dim: int, seed: int = SEED) -> list[Item]:
    rng = random.Random(seed)
    centers = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(CLUSTERS)]
    return [
        ([c + rng.gauss(0, SPREAD) for c in centers[rng.randrange(CLUSTERS)]], RID(i // 64, i % 64))
        for i in range(n)
    ]


def query_vectors(items: Sequence[Item], count: int, seed: int = SEED) -> list[list[float]]:
    """Perturbed copies of stored vectors: the query is near, but not equal to, a known one."""
    rng = random.Random(seed + 1)
    return [
        [v + rng.gauss(0, SPREAD / 4) for v in items[rng.randrange(len(items))][0]]
        for _ in range(count)
    ]


def nlist_for(size: int) -> int:
    # Regla habitual nlist ≈ sqrt(N): listas de ~sqrt(N) vectores, mismo costo de
    # elegir centroides que de recorrer las listas.
    return max(1, round(math.sqrt(size)))


class _Technique:
    """Build and reopen one technique in a scratch directory."""

    def __init__(self, name: str, metric: str, size: int, config: Config, root: Path) -> None:
        self.name = name
        self.metric = metric
        self.config = config
        self.path = root / f"{name}_{metric}_{size}"
        self.nlist = nlist_for(size)

    @property
    def params(self) -> str:
        if self.name == IVF:
            return f"nlist={self.nlist};nprobe={self.config.nprobe}"
        if self.name == HNSW:
            c = self.config
            return f"M={c.m};ef_construction={c.ef_construction};ef_search={c.ef_search}"
        return ""

    @property
    def query_opts(self) -> dict[str, object]:
        if self.name == IVF:
            return {"metric": self.metric, "nprobe": self.config.nprobe}
        if self.name == HNSW:
            return {"metric": self.metric, "ef": self.config.ef_search}
        return {"metric": self.metric}

    def build(self, items: Sequence[Item]) -> RankedIndex:
        c = self.config
        if self.name == BRUTE_FORCE:
            index = BruteForceIndex.open(
                self.path, c.dim, page_size=PAGE_SIZE, buffer_capacity=c.buffer_capacity
            )
            for vector, rid in items:
                index.insert(vector, rid)
            return index
        if self.name == IVF:
            return IVFIndex.build(
                self.path,
                items,
                dim=c.dim,
                nlist=self.nlist,
                seed=SEED,
                page_size=PAGE_SIZE,
                buffer_capacity=c.buffer_capacity,
            )
        index = HNSWIndex(
            self.path,
            c.dim,
            m=c.m,
            ef_construction=c.ef_construction,
            metric=self.metric,
            seed=SEED,
            page_size=PAGE_SIZE,
            buffer_capacity=c.buffer_capacity,
        )
        for vector, rid in items:
            index.insert(vector, rid)
        return index

    def reopen(self) -> RankedIndex:
        capacity = self.config.buffer_capacity
        if self.name == BRUTE_FORCE:
            return BruteForceIndex.open(self.path, page_size=PAGE_SIZE, buffer_capacity=capacity)
        if self.name == IVF:
            return IVFIndex.open(self.path, page_size=PAGE_SIZE, buffer_capacity=capacity)
        return HNSWIndex(self.path, page_size=PAGE_SIZE, buffer_capacity=capacity)


def _disks(index: RankedIndex) -> tuple[DiskManager, ...]:
    if isinstance(index, HNSWIndex):
        return index.disk_managers
    if isinstance(index, IVFIndex):
        return (index.disk_manager,)
    if isinstance(index, BruteForceIndex):
        return (index.store.disk_manager,)
    raise TypeError(f"unknown index {type(index).__name__}")


def _io(index: RankedIndex) -> tuple[int, int]:
    disks = _disks(index)
    return sum(d.reads for d in disks), sum(d.writes for d in disks)


def _bytes_on_disk(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.iterdir())
    return path.stat().st_size


def _measure_memory(action: Callable[[], object]) -> tuple[object, int, int]:
    """Run action under tracemalloc; return (result, retained bytes, peak bytes)."""
    tracemalloc.start()
    try:
        result = action()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, current, peak


def _recall(found: list[RID], expected: list[RID]) -> float:
    return len(set(found) & set(expected)) / len(expected) if expected else 1.0


def run(
    sizes: Sequence[int] = DATASET_SIZES,
    techniques: Sequence[str] = TECHNIQUES,
    metrics: Sequence[str] = METRICS,
    config: Config | None = None,
    progress: Callable[[str], None] = print,
) -> list[VectorResult]:
    config = config or Config()
    results: list[VectorResult] = []
    root = Path(tempfile.mkdtemp(prefix="vector_bench_"))
    try:
        for size in sizes:
            items = synthetic_vectors(size, config.dim)
            queries = query_vectors(items, config.queries)
            for metric in metrics:
                truth: list[list[RID]] | None = None
                # La fuerza bruta corre siempre y primero: sus resultados son el oráculo
                # de recall, aunque no se haya pedido reportarla.
                ordered = [BRUTE_FORCE, *(t for t in techniques if t != BRUTE_FORCE)]
                for name in ordered:
                    technique = _Technique(name, metric, size, config, root)
                    progress(f"{name} {metric} N={size}")
                    measured, found = _measure_technique(technique, items, queries, config, truth)
                    if truth is None:
                        truth = found
                    if name in techniques:
                        results.extend(measured)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return results


def _measure_technique(
    technique: _Technique,
    items: Sequence[Item],
    queries: Sequence[Sequence[float]],
    config: Config,
    truth: list[list[RID]] | None,
) -> tuple[list[VectorResult], list[list[RID]]]:
    size = len(items)

    def row(operation: str, run_number: int, **values: object) -> VectorResult:
        base = {
            "technique": technique.name,
            "operation": operation,
            "dataset_size": size,
            "run": run_number,
            "elapsed_ms": 0.0,
            "disk_reads": 0,
            "disk_writes": 0,
            "bytes_on_disk": 0,
            "metric": technique.metric,
            "dim": config.dim,
            "queries": len(queries),
            "k": config.k,
            "recall_at_k": 0.0,
            "memory_bytes": 0,
            "params": technique.params,
        }
        base.update(values)
        return VectorResult(**base)

    start = time.perf_counter()
    index = technique.build(items)
    build_ms = (time.perf_counter() - start) * 1000
    build_reads, build_writes = _io(index)
    index.close()

    reopened, retained, _peak = _measure_memory(technique.reopen)
    reopened.close()
    rows = [
        row(
            "build",
            1,
            elapsed_ms=build_ms,
            disk_reads=build_reads,
            disk_writes=build_writes,
            bytes_on_disk=_bytes_on_disk(technique.path),
            memory_bytes=retained,
        )
    ]

    found: list[list[RID]] = []
    for run_number in range(1, config.runs + 1):
        index = technique.reopen()
        reads_before, _ = _io(index)
        start = time.perf_counter()
        answers = [index.top_k(q, config.k, **technique.query_opts) for q in queries]
        elapsed = (time.perf_counter() - start) * 1000
        reads_after, _ = _io(index)
        index.close()
        found = [[rid for rid, _score in answer] for answer in answers]
        recall = (
            statistics.fmean(_recall(f, t) for f, t in zip(found, truth, strict=True))
            if truth is not None
            else 1.0
        )
        rows.append(
            row(
                "knn",
                run_number,
                elapsed_ms=elapsed / len(queries),
                disk_reads=(reads_after - reads_before) / len(queries),
                bytes_on_disk=_bytes_on_disk(technique.path),
                recall_at_k=recall,
            )
        )

    index = technique.reopen()
    _, _, peak = _measure_memory(
        lambda: [index.top_k(q, config.k, **technique.query_opts) for q in queries]
    )
    index.close()
    for result in rows[1:]:
        result.memory_bytes = peak
    return rows, found


def write_results(results: Sequence[VectorResult], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"vector_indexes_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(VectorResult)])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return path


def summarize(results: Sequence[VectorResult]) -> list[dict[str, object]]:
    """Median over runs per (technique, operation, metric, size), as the project reports."""
    groups: dict[tuple[str, str, str, int], list[VectorResult]] = {}
    for result in results:
        key = (result.technique, result.operation, result.metric, result.dataset_size)
        groups.setdefault(key, []).append(result)
    summary = []
    for (technique, operation, metric, size), rows in sorted(groups.items()):
        summary.append(
            {
                "technique": technique,
                "operation": operation,
                "metric": metric,
                "dataset_size": size,
                "elapsed_ms": statistics.median(r.elapsed_ms for r in rows),
                "disk_reads": statistics.median(r.disk_reads for r in rows),
                "recall_at_k": statistics.median(r.recall_at_k for r in rows),
                "bytes_on_disk": rows[0].bytes_on_disk,
                "memory_bytes": rows[0].memory_bytes,
            }
        )
    return summary


_COLORS = {BRUTE_FORCE: "#6b7280", IVF: "#2563eb", HNSW: "#dc2626"}


def time_vs_size_svg(results: Sequence[VectorResult], operation: str, title: str) -> str:
    """Log-log line chart of median time against dataset size; one line per technique·metric."""
    summary = [s for s in summarize(results) if s["operation"] == operation]
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for s in summary:
        series.setdefault((s["technique"], s["metric"]), []).append(
            (s["dataset_size"], max(s["elapsed_ms"], 1e-3))
        )
    width, height, pad = 640, 400, 60
    xs = [x for points in series.values() for x, _ in points] or [1]
    ys = [y for points in series.values() for _, y in points] or [1]
    x_lo, x_hi = math.log10(min(xs)), math.log10(max(xs))
    y_lo, y_hi = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))
    x_span = (x_hi - x_lo) or 1
    y_span = (y_hi - y_lo) or 1

    def px(x: float) -> float:
        return pad + (math.log10(x) - x_lo) / x_span * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (math.log10(y) - y_lo) / y_span * (height - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="15">{title}</text>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        'stroke="#111"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#111"/>',
        f'<text x="{width / 2}" y="{height - 15}" text-anchor="middle">registros (log)</text>',
        f'<text x="15" y="{height / 2}" transform="rotate(-90 15 {height / 2})" '
        'text-anchor="middle">ms (log)</text>',
    ]
    for size in sorted(set(xs)):
        parts.append(
            f'<text x="{px(size):.1f}" y="{height - pad + 18}" text-anchor="middle">{size}</text>'
        )
    for exponent in range(y_lo, y_hi + 1):
        y = py(10**exponent)
        parts.append(f'<text x="{pad - 8}" y="{y + 4:.1f}" text-anchor="end">1e{exponent}</text>')
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
    for position, ((technique, metric), points) in enumerate(sorted(series.items())):
        color = _COLORS.get(technique, "#000")
        dash = "" if metric == EUCLIDEAN else ' stroke-dasharray="6 4"'
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in sorted(points))
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
        legend_y = pad + 16 * position
        parts.append(
            f'<line x1="{width - pad - 150}" y1="{legend_y}" x2="{width - pad - 125}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="2"{dash}/>'
        )
        parts.append(
            f'<text x="{width - pad - 120}" y="{legend_y + 4}">{technique} · {metric}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DATASET_SIZES))
    parser.add_argument("--techniques", nargs="+", default=list(TECHNIQUES), choices=TECHNIQUES)
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--m", type=int, default=HNSW_M)
    parser.add_argument("--ef-construction", type=int, default=HNSW_EF_CONSTRUCTION)
    parser.add_argument("--ef-search", type=int, default=HNSW_EF_SEARCH)
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    config = Config(
        dim=args.dim,
        queries=args.queries,
        runs=args.runs,
        nprobe=args.nprobe,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    results = run(args.sizes, args.techniques, args.metrics, config)
    path = write_results(results, args.results_dir)
    for operation, title in (("knn", "Consulta k-NN (k=10)"), ("build", "Construcción")):
        svg = path.with_name(f"{path.stem}_{operation}.svg")
        svg.write_text(time_vs_size_svg(results, operation, title), encoding="utf-8")
    print(f"{len(results)} mediciones escritas en {path}")
    for line in summarize(results):
        print(
            f"{line['technique']:>12} {line['operation']:>5} {line['metric']:>9} "
            f"N={line['dataset_size']:>6} {line['elapsed_ms']:>10.2f} ms "
            f"reads={line['disk_reads']:>9.1f} recall={line['recall_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()
