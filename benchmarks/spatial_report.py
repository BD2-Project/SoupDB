"""Salida CSV de la comparación espacial, según el protocolo del informe (§6.3).

El `harness` general escribe una fila por operación con un solo tiempo. El
protocolo espacial exige media, mediana y p95 sobre 100 consultas, las dos
métricas de distancia y la memoria pico, así que esta suite usa su propio
esquema en vez de forzar el del harness y romper los otros benchmarks.

La columna `elapsed_ms` repite la mediana para que las herramientas que leen el
formato común —el panel de benchmarks del frontend— sigan funcionando.
"""

import csv
import statistics
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path


@dataclass
class SpatialResult:
    structure: str
    operation: str
    metric: str
    parameter: float
    records: int
    queries: int
    build_ms: float
    elapsed_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    disk_reads: int
    disk_writes: int
    size_bytes: int
    peak_rss_bytes: int
    result_count: int
    supported: bool = True


def percentile95(values: list[float]) -> float:
    """p95 por interpolación lineal, como `numpy.percentile` por defecto.

    No se usa `statistics.quantiles`: con n=100 devuelve los cortes de centiles y
    tomar el elemento 94 daría el p95 sin interpolar.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(samples: list[float]) -> tuple[float, float, float]:
    """(media, mediana, p95) de los tiempos por consulta."""
    if not samples:
        return 0.0, 0.0, 0.0
    return statistics.fmean(samples), statistics.median(samples), percentile95(samples)


def write_spatial_results(
    suite: str,
    results: list[SpatialResult],
    *,
    results_dir: str | Path = "benchmarks/results",
) -> Path:
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [field.name for field in fields(SpatialResult)]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return path
