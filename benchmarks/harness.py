"""Utilities for reproducible benchmarks with unified CSV output."""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """One measured benchmark operation."""

    structure: str
    operation: str
    records: int
    operations: int
    elapsed_ms: float
    ops_per_second: float
    disk_reads: int
    disk_writes: int
    size_bytes: int
    result_count: int
    supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_results(
    suite: str,
    results: list[BenchmarkResult],
    *,
    results_dir: str | Path = "benchmarks/results",
) -> Path:
    """Write benchmark results to a timestamped CSV file."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{suite}_{timestamp}.csv"

    fieldnames = list(BenchmarkResult.__dataclass_fields__)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.to_dict())

    return path
