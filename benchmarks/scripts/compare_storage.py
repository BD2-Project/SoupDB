"""Compare HeapFile vs SequentialFile: insert cost, primary-key scan lookup, and disk space."""

import argparse
import csv
import random
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from benchmarks.harness import BenchmarkResult, write_results
from engine.common.record import Record
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.heap_file import HeapFile
from engine.storage.sequential_file import SequentialFile

PAGE_SIZE = 4096
BUFFER_CAPACITY = 32
SEED = 42

KEY_SIZE = 8
RECORD_SIZE = 64
PAYLOAD_SIZE = RECORD_SIZE - KEY_SIZE

FileObj = HeapFile | SequentialFile
FileFactory = Callable[[Path], tuple[FileObj, DiskManager, BufferManager]]

RawRow = tuple[int, BenchmarkResult, int]


def make_record(key: int) -> Record:
    payload = bytes((key + i) % 256 for i in range(PAYLOAD_SIZE))
    return Record(data=key.to_bytes(KEY_SIZE, "big") + payload)


def record_key(record: Record) -> int:
    return int.from_bytes(record.data[:KEY_SIZE], "big")


def _rate(operations: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return operations / elapsed_seconds


def _open_heap(path: Path) -> tuple[HeapFile, DiskManager, BufferManager]:
    disk_manager = DiskManager(path, page_size=PAGE_SIZE)
    buffer_manager = BufferManager(disk_manager, capacity=BUFFER_CAPACITY)
    return HeapFile(disk_manager, buffer_manager), disk_manager, buffer_manager


def _open_sequential(path: Path) -> tuple[SequentialFile, DiskManager, BufferManager]:
    disk_manager = DiskManager(path, page_size=PAGE_SIZE)
    buffer_manager = BufferManager(disk_manager, capacity=BUFFER_CAPACITY)
    return (
        SequentialFile(disk_manager, buffer_manager, key_fn=record_key),
        disk_manager,
        buffer_manager,
    )


FACTORIES: dict[str, FileFactory] = {
    "heap": _open_heap,
    "sequential": _open_sequential,
}


def _benchmark_insert(
    structure: str,
    path: Path,
    keys: list[int],
    *,
    records: int,
) -> tuple[BenchmarkResult, int]:
    file_obj, disk_manager, buffer_manager = FACTORIES[structure](path)

    started = time.perf_counter()
    for key in keys:
        file_obj.insert(make_record(key))
    buffer_manager.flush_all()
    elapsed = time.perf_counter() - started

    result = BenchmarkResult(
        structure=structure,
        operation="insert",
        records=records,
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=disk_manager.reads,
        disk_writes=disk_manager.writes,
        size_bytes=path.stat().st_size,
        result_count=len(keys),
    )
    page_count = disk_manager.page_count
    disk_manager.close()
    return result, page_count


def _benchmark_search(
    structure: str,
    path: Path,
    target_keys: list[int],
    *,
    records: int,
) -> tuple[BenchmarkResult, int]:
    file_obj, disk_manager, _buffer_manager = FACTORIES[structure](path)

    reads_before = disk_manager.reads
    writes_before = disk_manager.writes

    started = time.perf_counter()
    result_count = 0

    for target in target_keys:
        for _rid, record in file_obj.scan():
            current_key = record_key(record)
            if current_key == target:
                result_count += 1
                break
            if structure == "sequential" and current_key > target:
                break

    elapsed = time.perf_counter() - started

    result = BenchmarkResult(
        structure=structure,
        operation="primary_key_scan_lookup",
        records=records,
        operations=len(target_keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(target_keys), elapsed),
        disk_reads=disk_manager.reads - reads_before,
        disk_writes=disk_manager.writes - writes_before,
        size_bytes=path.stat().st_size,
        result_count=result_count,
    )
    page_count = disk_manager.page_count
    disk_manager.close()
    return result, page_count


def _benchmark_reorganization(path: Path, *, records: int) -> tuple[BenchmarkResult, int]:
    """SequentialFile._reorganize_globally() was reverted from storage (see docs/decisiones.md);
    this reports the operation as unsupported instead of reimplementing it here."""
    file_obj, disk_manager, _buffer_manager = _open_sequential(path)

    result = BenchmarkResult(
        structure="sequential",
        operation="reorganization",
        records=records,
        operations=0,
        elapsed_ms=0.0,
        ops_per_second=0.0,
        disk_reads=0,
        disk_writes=0,
        size_bytes=path.stat().st_size,
        result_count=0,
        supported=hasattr(file_obj, "_reorganize_globally"),
    )
    page_count = disk_manager.page_count
    disk_manager.close()
    return result, page_count


def _median_result(rows: list[tuple[BenchmarkResult, int]]) -> tuple[BenchmarkResult, int]:
    sample = rows[0][0]
    aggregated = BenchmarkResult(
        structure=sample.structure,
        operation=sample.operation,
        records=sample.records,
        operations=sample.operations,
        elapsed_ms=statistics.median(r.elapsed_ms for r, _ in rows),
        ops_per_second=statistics.median(r.ops_per_second for r, _ in rows),
        disk_reads=round(statistics.median(r.disk_reads for r, _ in rows)),
        disk_writes=round(statistics.median(r.disk_writes for r, _ in rows)),
        size_bytes=round(statistics.median(r.size_bytes for r, _ in rows)),
        result_count=round(statistics.median(r.result_count for r, _ in rows)),
        supported=sample.supported,
    )
    page_count = round(statistics.median(p for _, p in rows))
    return aggregated, page_count


def run_benchmark(
    *,
    sizes: list[int],
    repetitions: int,
    query_count: int,
    seed: int,
) -> tuple[list[tuple[BenchmarkResult, int]], list[RawRow]]:
    aggregated: list[tuple[BenchmarkResult, int]] = []
    raw: list[RawRow] = []

    for record_count in sizes:
        keys = list(range(record_count))
        random.Random(seed).shuffle(keys)

        query_rng = random.Random(seed)
        target_keys = [query_rng.randrange(record_count) for _ in range(query_count)]

        per_operation: dict[tuple[str, str], list[tuple[BenchmarkResult, int]]] = defaultdict(list)

        for repetition in range(repetitions):
            with tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)

                for structure in ("heap", "sequential"):
                    path = workdir / f"{structure}.db"

                    insert_result = _benchmark_insert(structure, path, keys, records=record_count)
                    per_operation[(structure, "insert")].append(insert_result)
                    raw.append((repetition, *insert_result))

                    search_result = _benchmark_search(
                        structure, path, target_keys, records=record_count
                    )
                    per_operation[(structure, "primary_key_scan_lookup")].append(search_result)
                    raw.append((repetition, *search_result))

                    if structure == "sequential":
                        reorg_result = _benchmark_reorganization(path, records=record_count)
                        per_operation[(structure, "reorganization")].append(reorg_result)
                        raw.append((repetition, *reorg_result))

        for rows in per_operation.values():
            aggregated.append(_median_result(rows))

    return aggregated, raw


def _print_results(rows: list[tuple[BenchmarkResult, int]]) -> None:
    header = (
        f"{'records':>10} "
        f"{'structure':<12} "
        f"{'operation':<26} "
        f"{'ms':>12} "
        f"{'ops/s':>14} "
        f"{'reads':>10} "
        f"{'writes':>10} "
        f"{'bytes':>12} "
        f"{'pages':>8} "
        f"{'results':>10}"
    )
    print(header)
    print("-" * len(header))

    for result, page_count in rows:
        operation = result.operation
        if not result.supported:
            operation += " (N/A)"

        print(
            f"{result.records:>10} "
            f"{result.structure:<12} "
            f"{operation:<26} "
            f"{result.elapsed_ms:>12.3f} "
            f"{result.ops_per_second:>14.2f} "
            f"{result.disk_reads:>10} "
            f"{result.disk_writes:>10} "
            f"{result.size_bytes:>12} "
            f"{page_count:>8} "
            f"{result.result_count:>10}"
        )


def _write_raw_csv(rows: list[RawRow], *, results_dir: str | Path = "benchmarks/results") -> Path:
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"storage_raw_{timestamp}.csv"

    fieldnames = ["repetition", *BenchmarkResult.__dataclass_fields__, "page_count"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for repetition, result, page_count in rows:
            row = result.to_dict()
            row["repetition"] = repetition
            row["page_count"] = page_count
            writer.writerow(row)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HeapFile vs SequentialFile.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10_000, 100_000])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    aggregated, raw = run_benchmark(
        sizes=args.sizes,
        repetitions=args.repetitions,
        query_count=args.queries,
        seed=args.seed,
    )

    print("operation 'primary_key_scan_lookup': linear scan(), not an index or binary search.\n")

    _print_results(aggregated)

    csv_path = write_results("storage", [result for result, _ in aggregated])
    raw_path = _write_raw_csv(raw)

    print(f"\nCSV: {csv_path}")
    print(f"Raw CSV: {raw_path}")


if __name__ == "__main__":
    main()
