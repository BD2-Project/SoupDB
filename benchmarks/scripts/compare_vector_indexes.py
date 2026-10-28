"""Comparación experimental 2.4.4: IVF contra HNSW, euclidiana contra coseno.

Las tres técnicas (fuerza bruta, IVF y HNSW) se miden con los mismos vectores
y las mismas consultas, con las dos métricas. La fuerza bruta es además el
oráculo del Recall@k, con una verdad exacta independiente por métrica.

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
- Sintético (por defecto): gaussianas con semilla alrededor de centros al azar;
  las consultas son copias perturbadas de vectores guardados.
- Real (``--real-data-dir``): los ``vectores.bin`` de imágenes (COCO) y de
  segmentos de audio (FMA). Las consultas son ``queries`` objetos del propio
  dataset elegidos con ``SEED``. Como ya están en el índice, a cada técnica se
  le piden ``k + 1`` vecinos y se descarta el propio objeto (:func:`knn_ids`):
  la misma regla para fuerza bruta, IVF y HNSW.

Con vectores L2-normalizados (los TF-IDF reales lo están) ``|a-b|² = 2 - 2·cos``,
así que euclidiana y coseno ordenan igual: si sus resultados coinciden, es lo
esperado y no un error.

Uso::

    uv run python -m benchmarks.scripts.compare_vector_indexes
    uv run python -m benchmarks.scripts.compare_vector_indexes --sizes 1000 10000 100000
    uv run python -m benchmarks.scripts.compare_vector_indexes \\
        --real-data-dir D:/SoupDB-data/multimedia --sizes 1000 10000 100000 \\
        --image-root D:/datasets/coco/train2017 --audio-root D:/datasets/fma_medium
"""

import argparse
import csv
import html
import json
import math
import random
import shutil
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.vector.brute_force import BruteForceIndex
from engine.indexes.vector.constants import HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M
from engine.indexes.vector.hnsw import HNSWIndex
from engine.indexes.vector.ivf import IVFIndex
from engine.indexes.vector.metrics import COSINE, EUCLIDEAN
from engine.multimedia.vector_file import iter_vectors
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

SYNTHETIC = "synthetic"
IMAGE = "image"
AUDIO = "audio"
MEDIA_TYPES = (IMAGE, AUDIO)
REAL_SIZES = (1_000, 10_000, 100_000)
#: Subcarpeta (y prefijo de manifest) de cada medio dentro de --real-data-dir.
MEDIA_DIRS = {IMAGE: "images", AUDIO: "audio"}
#: Consultas de cada dataset que se muestran en la galería (las de más errores).
GALLERY_QUERIES = 3
RIDS_PER_PAGE = 64

Item = tuple[Sequence[float], RID]


@dataclass
class VectorResult:
    media_type: str
    dataset_size: int
    technique: str
    metric: str
    operation: str
    run: int
    elapsed_ms: float
    disk_reads: float
    disk_writes: int
    bytes_on_disk: int
    memory_bytes: int
    recall_at_k: float
    dim: int
    queries: int
    k: int
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


class ObjectInfo(NamedTuple):
    """Metadatos originales de un objeto: id y path de vectores.bin, y su medio."""

    id: int
    path: str
    media_type: str
    #: Intervalo en segundos de un segmento de audio (del manifest), si se conoce.
    start: float | None = None
    end: float | None = None


@dataclass
class Dataset:
    """Vectores, consultas y metadatos de un tamaño; idénticos para todas las técnicas."""

    media_type: str
    size: int
    dim: int
    #: (vector, RID) reiterable: IVF lo recorre dos veces (muestra y carga).
    items: Iterable[Item]
    queries: list[Sequence[float]]
    #: RID de cada consulta si es un objeto del índice: se excluye de su propia respuesta.
    query_rids: list[RID | None]
    metadata: dict[RID, ObjectInfo] = field(default_factory=dict)


def rid_for(position: int) -> RID:
    # RID por posición ordinal: no depende de la técnica ni de la métrica.
    return RID(position // RIDS_PER_PAGE, position % RIDS_PER_PAGE)


def synthetic_vectors(n: int, dim: int, seed: int = SEED) -> list[Item]:
    rng = random.Random(seed)
    centers = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(CLUSTERS)]
    return [
        ([c + rng.gauss(0, SPREAD) for c in centers[rng.randrange(CLUSTERS)]], rid_for(i))
        for i in range(n)
    ]


def query_vectors(items: Sequence[Item], count: int, seed: int = SEED) -> list[list[float]]:
    """Perturbed copies of stored vectors: the query is near, but not equal to, a known one."""
    rng = random.Random(seed + 1)
    return [
        [v + rng.gauss(0, SPREAD / 4) for v in items[rng.randrange(len(items))][0]]
        for _ in range(count)
    ]


def synthetic_dataset(size: int, config: Config) -> Dataset:
    items = synthetic_vectors(size, config.dim)
    queries: list[Sequence[float]] = list(query_vectors(items, config.queries))
    # Las consultas sintéticas no son objetos del índice: no hay nada que excluir.
    return Dataset(SYNTHETIC, size, config.dim, items, queries, [None] * len(queries))


class VectorFileItems:
    """(vector, RID) de un vectores.bin; cada recorrido vuelve a leer el archivo.

    Ninguna técnica necesita los N vectores en memoria y todas reciben los
    mismos vectores con los mismos RIDs.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[Item]:
        for position, record in enumerate(iter_vectors(self.path)):
            yield record.vector, rid_for(position)


def select_query_positions(size: int, count: int, seed: int = SEED) -> list[int]:
    """Posiciones de los objetos consulta; las mismas para toda técnica y métrica."""
    return sorted(random.Random(seed).sample(range(size), min(count, size)))


def load_real_dataset(
    vectors_path: str | Path, media_type: str, queries: int = QUERIES, seed: int = SEED
) -> Dataset:
    """Lee vectores.bin: metadatos RID -> objeto, dimensión y vectores de las consultas."""
    metadata: dict[RID, ObjectInfo] = {}
    dim = 0
    for position, record in enumerate(iter_vectors(vectors_path)):
        dim = len(record.vector)
        metadata[rid_for(position)] = ObjectInfo(record.id, record.path, media_type)
    if not metadata:
        raise ValueError(f"{vectors_path} has no vectors")

    positions = select_query_positions(len(metadata), queries, seed)
    wanted = set(positions)
    vectors = {
        position: record.vector
        for position, record in enumerate(iter_vectors(vectors_path))
        if position in wanted
    }
    return Dataset(
        media_type,
        len(metadata),
        dim,
        VectorFileItems(vectors_path),
        [vectors[p] for p in positions],
        [rid_for(p) for p in positions],
        metadata,
    )


def knn_ids(
    index: RankedIndex, query: Sequence[float], k: int, exclude: RID | None, **opts: object
) -> list[RID]:
    """Top-k de una consulta; si la consulta es un objeto del índice, sin él mismo.

    Única regla para las tres técnicas: se piden k + 1, se quita el RID de la
    consulta (esté donde esté) y se conservan los k primeros.
    """
    if exclude is None:
        return [rid for rid, _score in index.top_k(query, k, **opts)]
    answer = index.top_k(query, k + 1, **opts)
    return [rid for rid, _score in answer if rid != exclude][:k]


def recall_at_k(found: Sequence[RID], exact: Sequence[RID]) -> float:
    """|top-k aproximado ∩ top-k exacto| / k, con el exacto de la misma métrica."""
    return len(set(found) & set(exact)) / len(exact) if exact else 1.0


def nlist_for(size: int) -> int:
    # Regla habitual nlist ≈ sqrt(N): listas de ~sqrt(N) vectores, mismo costo de
    # elegir centroides que de recorrer las listas.
    return max(1, round(math.sqrt(size)))


class _Technique:
    """Build and reopen one technique in a scratch directory."""

    def __init__(
        self, name: str, metric: str, dataset: Dataset, config: Config, root: Path
    ) -> None:
        self.name = name
        self.metric = metric
        self.config = config
        self.dim = dataset.dim
        self.path = root / f"{dataset.media_type}_{name}_{metric}_{dataset.size}"
        self.nlist = nlist_for(dataset.size)

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

    def build(self, items: Iterable[Item]) -> RankedIndex:
        c = self.config
        if self.name == BRUTE_FORCE:
            index = BruteForceIndex.open(
                self.path, self.dim, page_size=PAGE_SIZE, buffer_capacity=c.buffer_capacity
            )
            for vector, rid in items:
                index.insert(vector, rid)
            return index
        if self.name == IVF:
            return IVFIndex.build(
                self.path,
                items,
                dim=self.dim,
                nlist=self.nlist,
                seed=SEED,
                page_size=PAGE_SIZE,
                buffer_capacity=c.buffer_capacity,
            )
        # HNSW guarda la métrica con la que se construye: un grafo distinto por métrica.
        index = HNSWIndex(
            self.path,
            self.dim,
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


#: Respuestas de la última corrida por (métrica, técnica): una lista de RIDs por consulta.
Answers = dict[tuple[str, str], list[list[RID]]]


def run(
    sizes: Sequence[int] = DATASET_SIZES,
    techniques: Sequence[str] = TECHNIQUES,
    metrics: Sequence[str] = METRICS,
    config: Config | None = None,
    progress: Callable[[str], None] = print,
) -> list[VectorResult]:
    """Benchmark sintético."""
    config = config or Config()
    datasets = (synthetic_dataset(size, config) for size in sizes)
    return run_datasets(datasets, techniques, metrics, config, progress)


def run_datasets(
    datasets: Iterable[Dataset],
    techniques: Sequence[str] = TECHNIQUES,
    metrics: Sequence[str] = METRICS,
    config: Config | None = None,
    progress: Callable[[str], None] = print,
    on_answers: Callable[[Dataset, Answers], None] | None = None,
) -> list[VectorResult]:
    """Mide cada dataset (se cargan de uno en uno) con todas las técnicas y métricas."""
    config = config or Config()
    results: list[VectorResult] = []
    root = Path(tempfile.mkdtemp(prefix="vector_bench_"))
    try:
        for dataset in datasets:
            answers: Answers = {}
            for metric in metrics:
                truth: list[list[RID]] | None = None
                # La fuerza bruta corre siempre y primero: sus resultados son el oráculo
                # de recall de esta métrica, aunque no se haya pedido reportarla.
                ordered = [BRUTE_FORCE, *(t for t in techniques if t != BRUTE_FORCE)]
                for name in ordered:
                    technique = _Technique(name, metric, dataset, config, root)
                    progress(f"{dataset.media_type} {name} {metric} N={dataset.size}")
                    measured, found = _measure_technique(technique, dataset, config, truth)
                    answers[(metric, name)] = found
                    if truth is None:
                        truth = found
                    if name in techniques:
                        results.extend(measured)
                    # Un índice de 100k ocupa cientos de MB: se borra al terminar de medirlo.
                    shutil.rmtree(technique.path, ignore_errors=True)
            if on_answers is not None:
                on_answers(dataset, answers)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return results


def _measure_technique(
    technique: _Technique,
    dataset: Dataset,
    config: Config,
    truth: list[list[RID]] | None,
) -> tuple[list[VectorResult], list[list[RID]]]:
    queries = dataset.queries
    pairs = list(zip(queries, dataset.query_rids, strict=True))

    def answer_all(index: RankedIndex) -> list[list[RID]]:
        opts = technique.query_opts
        return [knn_ids(index, query, config.k, exclude, **opts) for query, exclude in pairs]

    def row(operation: str, run_number: int, **values: object) -> VectorResult:
        base = {
            "media_type": dataset.media_type,
            "dataset_size": dataset.size,
            "technique": technique.name,
            "metric": technique.metric,
            "operation": operation,
            "run": run_number,
            "elapsed_ms": 0.0,
            "disk_reads": 0,
            "disk_writes": 0,
            "bytes_on_disk": 0,
            "memory_bytes": 0,
            "recall_at_k": 0.0,
            "dim": dataset.dim,
            "queries": len(queries),
            "k": config.k,
            "params": technique.params,
        }
        base.update(values)
        return VectorResult(**base)

    start = time.perf_counter()
    index = technique.build(dataset.items)
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
        found = answer_all(index)
        elapsed = (time.perf_counter() - start) * 1000
        reads_after, _ = _io(index)
        index.close()
        recall = (
            statistics.fmean(recall_at_k(f, t) for f, t in zip(found, truth, strict=True))
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
    _, _, peak = _measure_memory(lambda: answer_all(index))
    index.close()
    for result in rows[1:]:
        result.memory_bytes = peak
    return rows, found


def results_path(results_dir: Path, prefix: str = "vector_indexes") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"


def append_results(path: Path, results: Sequence[VectorResult]) -> None:
    """Añade filas al CSV (con cabecera si aún no existe); usado tras cada dataset."""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(VectorResult)])
        if new:
            writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_results(
    results: Sequence[VectorResult], results_dir: Path, prefix: str = "vector_indexes"
) -> Path:
    path = results_path(results_dir, prefix)
    append_results(path, results)
    return path


def summarize(results: Sequence[VectorResult]) -> list[dict[str, object]]:
    """Median over runs per (media, technique, operation, metric, size)."""
    groups: dict[tuple[str, str, str, str, int], list[VectorResult]] = {}
    for result in results:
        key = (
            result.media_type,
            result.technique,
            result.operation,
            result.metric,
            result.dataset_size,
        )
        groups.setdefault(key, []).append(result)
    summary = []
    for (media_type, technique, operation, metric, size), rows in sorted(groups.items()):
        summary.append(
            {
                "media_type": media_type,
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


def time_vs_size_svg(
    results: Sequence[VectorResult], operation: str, title: str, media_type: str | None = None
) -> str:
    """Log-log line chart of median time against dataset size; one line per technique·metric.

    Color = técnica; trazo continuo = euclidiana, discontinuo = coseno. Con
    ``media_type`` solo entra ese medio, para no mezclar imagen y audio.
    """
    if media_type is not None:
        results = [r for r in results if r.media_type == media_type]
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


_OPERATION_TITLES = {"knn": "Consulta k-NN (k=10)", "build": "Construcción"}
_MEDIA_TITLES = {SYNTHETIC: "sintético", IMAGE: "imagen", AUDIO: "audio"}


def write_charts(results: Sequence[VectorResult], csv_path: Path) -> list[Path]:
    """Un SVG por (medio, operación): ``<csv>_<medio>_<build|knn>.svg``."""
    paths = []
    for media_type in sorted({r.media_type for r in results}):
        for operation, title in _OPERATION_TITLES.items():
            svg = csv_path.with_name(f"{csv_path.stem}_{media_type}_{operation}.svg")
            label = f"{title} · {_MEDIA_TITLES.get(media_type, media_type)}"
            svg.write_text(time_vs_size_svg(results, operation, label, media_type), "utf-8")
            paths.append(svg)
    return paths


# --- datos reales: carga por dataset y galería ----------------------------------------


@dataclass
class GalleryQuery:
    media_type: str
    size: int
    query: ObjectInfo
    #: (métrica, técnica) -> top-k de esa técnica para esta consulta.
    results: dict[tuple[str, str], list[ObjectInfo]]


def _size_label(size: int) -> str:
    return f"{size // 1000}k" if size % 1000 == 0 else str(size)


def vectors_for(data_dir: str | Path, media_type: str, size: int) -> Path:
    return Path(data_dir) / MEDIA_DIRS[media_type] / str(size) / "vectores.bin"


def manifest_for(data_dir: str | Path, media_type: str, size: int) -> Path:
    name = f"{MEDIA_DIRS[media_type]}_{_size_label(size)}.jsonl"
    return Path(data_dir) / "manifests" / name


def _with_intervals(infos: dict[int, ObjectInfo], manifest: Path) -> dict[int, ObjectInfo]:
    """Añade start/end (s) a los segmentos de audio, leídos del manifest por id."""
    if not manifest.is_file():
        return infos
    enriched = dict(infos)
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            info = enriched.get(row["id"])
            if info is not None and "start" in row:
                enriched[row["id"]] = info._replace(start=row["start"], end=row["end"])
    return enriched


def gallery_queries(
    dataset: Dataset, answers: Answers, count: int, manifest: Path | None = None
) -> list[GalleryQuery]:
    """``count`` consultas con el top-k de cada técnica, priorizando las que tienen errores.

    Determinista: se ordenan por número de MISS de IVF/HNSW (sumando métricas) y, a
    igualdad, por posición; si no hay errores quedan las primeras. Se muestran en
    orden de posición.
    """

    def misses(i: int) -> int:
        total = 0
        for (metric, technique), found in answers.items():
            exact = answers.get((metric, BRUTE_FORCE))
            if technique != BRUTE_FORCE and exact is not None:
                total += len(set(found[i]) - set(exact[i]))
        return total

    ranked = sorted(range(len(dataset.queries)), key=lambda i: (-misses(i), i))
    chosen = sorted(ranked[:count])
    needed = {dataset.query_rids[i] for i in chosen}
    needed |= {rid for found in answers.values() for i in chosen for rid in found[i]}
    needed.discard(None)
    by_id = {dataset.metadata[rid].id: dataset.metadata[rid] for rid in needed}
    if manifest is not None and dataset.media_type == AUDIO:
        by_id = _with_intervals(by_id, manifest)

    def info(rid: RID) -> ObjectInfo:
        return by_id[dataset.metadata[rid].id]

    return [
        GalleryQuery(
            dataset.media_type,
            dataset.size,
            info(dataset.query_rids[i]),
            {key: [info(rid) for rid in found[i]] for key, found in answers.items()},
        )
        for i in chosen
    ]


def run_real(
    data_dir: str | Path,
    media_types: Sequence[str] = MEDIA_TYPES,
    sizes: Sequence[int] = REAL_SIZES,
    techniques: Sequence[str] = TECHNIQUES,
    metrics: Sequence[str] = METRICS,
    config: Config | None = None,
    progress: Callable[[str], None] = print,
    gallery_count: int = GALLERY_QUERIES,
) -> tuple[list[VectorResult], list[GalleryQuery]]:
    """Benchmark sobre ``<data_dir>/{images,audio}/<N>/vectores.bin``, un dataset a la vez."""
    config = config or Config()
    gallery: list[GalleryQuery] = []

    def datasets() -> Iterator[Dataset]:
        for media_type in media_types:
            for size in sizes:
                path = vectors_for(data_dir, media_type, size)
                progress(f"cargando {path}")
                yield load_real_dataset(path, media_type, config.queries)

    def collect(dataset: Dataset, answers: Answers) -> None:
        manifest = manifest_for(data_dir, dataset.media_type, dataset.size)
        gallery.extend(gallery_queries(dataset, answers, gallery_count, manifest))

    results = run_datasets(datasets(), techniques, metrics, config, progress, collect)
    return results, gallery


def _split_segment(path: str) -> tuple[str, int | None]:
    # "000/000002.mp3#segment=000003" -> ("000/000002.mp3", 3)
    file, marker, segment = path.partition("#segment=")
    return file, int(segment) if marker else None


def _media_html(info: ObjectInfo, root: Path | None) -> str:
    file, segment = _split_segment(info.path)
    caption = f"id {info.id} · {html.escape(file)}"
    if segment is not None:
        caption += f" · segmento {segment}"
        if info.start is not None:
            caption += f" · {info.start:g}–{info.end:g} s"
    caption_html = f'<div class="cap">{caption}</div>'
    if root is None:
        return caption_html
    uri = (Path(root) / file).resolve().as_uri()
    if info.media_type == IMAGE:
        return f'<img loading="lazy" src="{html.escape(uri)}" alt="">{caption_html}'
    # Media fragment #t=inicio,fin: el reproductor suena solo ese segmento del MP3.
    if info.start is not None:
        uri += f"#t={info.start:g},{info.end:g}"
    return f'<audio controls preload="none" src="{html.escape(uri)}"></audio>{caption_html}'


_GALLERY_CSS = """
:root { --bg:#fff; --fg:#111827; --muted:#6b7280; --line:#e5e7eb;
        --hit:#15803d; --miss:#b91c1c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111827; --fg:#f3f4f6; --muted:#9ca3af; --line:#374151;
          --hit:#4ade80; --miss:#f87171; }
}
body { background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif;
       margin:0 auto; padding:16px; max-width:1500px; }
h2 { margin-top:40px; border-top:2px solid var(--line); padding-top:16px; }
.row { display:flex; gap:8px; overflow-x:auto; padding:4px 0 12px; align-items:flex-start; }
.label { flex:0 0 150px; font-weight:600; }
.label small { display:block; font-weight:400; color:var(--muted); }
.card { flex:0 0 150px; border:2px solid var(--line); border-radius:6px; padding:4px; }
.card img { width:100%; height:110px; object-fit:cover; display:block; }
.card audio { width:100%; }
.card.hit { border-color:var(--hit); }
.card.miss { border-color:var(--miss); }
.badge { font-weight:700; font-size:12px; }
.hit .badge { color:var(--hit); }
.miss .badge { color:var(--miss); }
.cap { font-size:11px; color:var(--muted); word-break:break-all; }
.query { max-width:340px; }
"""


def render_gallery(
    entries: Sequence[GalleryQuery],
    roots: dict[str, Path | None],
    metrics: Sequence[str] = METRICS,
    k: int = K,
) -> str:
    """HTML con cada consulta, el top-k exacto y el de IVF/HNSW marcado HIT/MISS."""
    parts = [
        '<!doctype html><html lang="es"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Galería k-NN</title><style>{_GALLERY_CSS}</style></head><body>",
        "<h1>Galería k-NN: aciertos y errores</h1>",
        f"<p>Top-{k} sin el propio objeto consulta. HIT: está en el top-{k} exacto de la "
        "fuerza bruta con la misma métrica. MISS: no está.</p>",
    ]
    for entry in entries:
        root = roots.get(entry.media_type)
        parts.append(
            f"<h2>{entry.media_type} · N={entry.size} · consulta id {entry.query.id}</h2>"
            f'<div class="card query"><div class="badge">consulta</div>'
            f"{_media_html(entry.query, root)}</div>"
        )
        for metric in metrics:
            exact = entry.results.get((metric, BRUTE_FORCE))
            if exact is None:
                continue
            exact_ids = {info.id for info in exact}
            parts.append(f"<h3>{metric}</h3>")
            for technique in TECHNIQUES:
                found = entry.results.get((metric, technique))
                if found is None:
                    continue
                if technique == BRUTE_FORCE:
                    detail = f"exacto · {metric} · N={entry.size}"
                else:
                    hits = sum(info.id in exact_ids for info in found)
                    detail = f"{metric} · N={entry.size} · recall {hits}/{len(exact)}"
                cards = []
                for rank, info in enumerate(found, start=1):
                    if technique == BRUTE_FORCE:
                        status, badge = "", f"#{rank}"
                    elif info.id in exact_ids:
                        status, badge = " hit", f"#{rank} HIT"
                    else:
                        status, badge = " miss", f"#{rank} MISS"
                    cards.append(
                        f'<div class="card{status}"><div class="badge">{badge}</div>'
                        f"{_media_html(info, root)}</div>"
                    )
                parts.append(
                    f'<div class="row"><div class="label">{technique}<small>{detail}</small>'
                    f"</div>{''.join(cards)}</div>"
                )
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=None)
    parser.add_argument("--techniques", nargs="+", default=list(TECHNIQUES), choices=TECHNIQUES)
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--dim", type=int, default=DIM, help="solo sintético; real usa la de datos")
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--m", type=int, default=HNSW_M)
    parser.add_argument("--ef-construction", type=int, default=HNSW_EF_CONSTRUCTION)
    parser.add_argument("--ef-search", type=int, default=HNSW_EF_SEARCH)
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    real = parser.add_argument_group("datos reales")
    real.add_argument("--real-data-dir", type=Path, help="carpeta con images/, audio/, manifests/")
    real.add_argument("--media", nargs="+", default=list(MEDIA_TYPES), choices=MEDIA_TYPES)
    real.add_argument("--image-root", type=Path, help="COCO train2017 (imágenes de la galería)")
    real.add_argument("--audio-root", type=Path, help="fma_medium (audios de la galería)")
    real.add_argument("--gallery-queries", type=int, default=GALLERY_QUERIES)
    args = parser.parse_args(argv)

    config = Config(
        dim=args.dim,
        queries=args.queries,
        runs=args.runs,
        nprobe=args.nprobe,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    if args.real_data_dir is None:
        results = run(args.sizes or list(DATASET_SIZES), args.techniques, args.metrics, config)
        path = write_results(results, args.results_dir)
    else:
        # Un dataset a la vez y su CSV al terminar: si una ejecución de horas se corta,
        # las filas de los datasets ya medidos quedan escritas.
        path = results_path(args.results_dir, prefix="vector_indexes_real")
        results, gallery = [], []
        for media_type in args.media:
            for size in args.sizes or list(REAL_SIZES):
                measured, queries = run_real(
                    args.real_data_dir,
                    [media_type],
                    [size],
                    args.techniques,
                    args.metrics,
                    config,
                    gallery_count=args.gallery_queries,
                )
                append_results(path, measured)
                results.extend(measured)
                gallery.extend(queries)
                print(f"{media_type} N={size}: {len(measured)} filas añadidas a {path}")
        page = path.with_name(f"{path.stem}_gallery.html")
        roots = {IMAGE: args.image_root, AUDIO: args.audio_root}
        page.write_text(render_gallery(gallery, roots, args.metrics, config.k), "utf-8")
        print(f"galería: {page}")
    for svg in write_charts(results, path):
        print(f"gráfica: {svg}")
    print(f"{len(results)} mediciones escritas en {path}")
    for line in summarize(results):
        print(
            f"{line['media_type']:>9} {line['technique']:>12} {line['operation']:>5} "
            f"{line['metric']:>9} N={line['dataset_size']:>6} {line['elapsed_ms']:>10.2f} ms "
            f"reads={line['disk_reads']:>9.1f} recall={line['recall_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()
