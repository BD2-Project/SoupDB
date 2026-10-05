"""Compare B+ tree and extendible-hash indexes under equivalent workloads."""

import argparse
import random
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from benchmarks.harness import BenchmarkResult, write_results
from engine.common.rid import RID
from engine.indexes.base import Index
from engine.indexes.bplus_tree import BPlusTree
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 4096
BUFFER_CAPACITY = 32

IndexFactory = Callable[[DiskManager, BufferManager], Index]


def _bplus_factory(
    disk_manager: DiskManager,
    buffer_manager: BufferManager,
) -> Index:
    return BPlusTree(disk_manager, buffer_manager)


def _hash_factory(
    disk_manager: DiskManager,
    buffer_manager: BufferManager,
) -> Index:
    return ExtendibleHash(disk_manager, buffer_manager)


STRUCTURES: dict[str, IndexFactory] = {
    "bplus": _bplus_factory,
    "extendible_hash": _hash_factory,
}


def _rid(number: int) -> RID:
    return RID(number // 128, number % 128)


def _rate(operations: int, elapsed_seconds: float) -> float:
    if elapsed_seconds == 0:
        return 0.0
    return operations / elapsed_seconds


def _build_index(
    structure: str,
    factory: IndexFactory,
    keys: list[int],
    path: Path,
) -> BenchmarkResult:
    disk_manager = DiskManager(path, page_size=PAGE_SIZE)
    buffer_manager = BufferManager(
        disk_manager,
        capacity=BUFFER_CAPACITY,
    )
    index = factory(disk_manager, buffer_manager)

    reads_before = disk_manager.reads
    writes_before = disk_manager.writes

    started = time.perf_counter()

    for position, key in enumerate(keys):
        index.insert(key, _rid(position))

    index.close()
    elapsed = time.perf_counter() - started

    reads = disk_manager.reads - reads_before
    writes = disk_manager.writes - writes_before

    disk_manager.close()

    return BenchmarkResult(
        structure=structure,
        operation="build",
        records=len(keys),
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=reads,
        disk_writes=writes,
        size_bytes=path.stat().st_size,
        result_count=len(keys),
    )


def _open_index(
    path: Path,
    factory: IndexFactory,
) -> tuple[Index, DiskManager]:
    disk_manager = DiskManager(path, page_size=PAGE_SIZE)
    buffer_manager = BufferManager(
        disk_manager,
        capacity=BUFFER_CAPACITY,
    )
    return factory(disk_manager, buffer_manager), disk_manager


def _benchmark_equality(
    structure: str,
    factory: IndexFactory,
    source: Path,
    keys: list[int],
    *,
    records: int,
) -> BenchmarkResult:
    index, disk_manager = _open_index(source, factory)

    started = time.perf_counter()
    result_count = 0

    for key in keys:
        result_count += len(index.search(key))

    elapsed = time.perf_counter() - started

    result = BenchmarkResult(
        structure=structure,
        operation="equality_search",
        records=records,
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=disk_manager.reads,
        disk_writes=disk_manager.writes,
        size_bytes=source.stat().st_size,
        result_count=result_count,
    )

    index.close()
    disk_manager.close()
    return result


def _benchmark_range(
    structure: str,
    factory: IndexFactory,
    source: Path,
    ranges: list[tuple[int, int]],
    *,
    records: int,
) -> BenchmarkResult:
    index, disk_manager = _open_index(source, factory)

    if not index.supports_range:
        result = BenchmarkResult(
            structure=structure,
            operation="range_search",
            records=records,
            operations=len(ranges),
            elapsed_ms=0.0,
            ops_per_second=0.0,
            disk_reads=0,
            disk_writes=0,
            size_bytes=source.stat().st_size,
            result_count=0,
            supported=False,
        )
        index.close()
        disk_manager.close()
        return result

    started = time.perf_counter()
    result_count = 0

    for lo, hi in ranges:
        result_count += len(index.range_search(lo, hi))

    elapsed = time.perf_counter() - started

    result = BenchmarkResult(
        structure=structure,
        operation="range_search",
        records=records,
        operations=len(ranges),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(ranges), elapsed),
        disk_reads=disk_manager.reads,
        disk_writes=disk_manager.writes,
        size_bytes=source.stat().st_size,
        result_count=result_count,
    )

    index.close()
    disk_manager.close()
    return result


def _benchmark_insert(
    structure: str,
    factory: IndexFactory,
    source: Path,
    working: Path,
    keys: list[int],
    *,
    records: int,
) -> BenchmarkResult:
    shutil.copyfile(source, working)
    index, disk_manager = _open_index(working, factory)

    reads_before = disk_manager.reads
    writes_before = disk_manager.writes

    started = time.perf_counter()

    for offset, key in enumerate(keys):
        index.insert(key, _rid(records + offset))

    index.close()
    elapsed = time.perf_counter() - started

    result = BenchmarkResult(
        structure=structure,
        operation="insert",
        records=records,
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=disk_manager.reads - reads_before,
        disk_writes=disk_manager.writes - writes_before,
        size_bytes=working.stat().st_size,
        result_count=len(keys),
    )

    disk_manager.close()
    return result


def _benchmark_remove(
    structure: str,
    factory: IndexFactory,
    source: Path,
    working: Path,
    keys: list[int],
    rid_by_key: dict[int, RID],
    *,
    records: int,
) -> BenchmarkResult:
    shutil.copyfile(source, working)
    index, disk_manager = _open_index(working, factory)

    reads_before = disk_manager.reads
    writes_before = disk_manager.writes

    started = time.perf_counter()
    removed = 0

    for key in keys:
        removed += index.remove(key, rid_by_key[key])

    index.close()
    elapsed = time.perf_counter() - started

    result = BenchmarkResult(
        structure=structure,
        operation="remove",
        records=records,
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=disk_manager.reads - reads_before,
        disk_writes=disk_manager.writes - writes_before,
        size_bytes=working.stat().st_size,
        result_count=removed,
    )

    disk_manager.close()
    return result


def run_benchmark(
    *,
    record_count: int,
    query_count: int,
    mutation_count: int,
    seed: int,
) -> list[BenchmarkResult]:
    if record_count <= 0:
        raise ValueError("record_count must be positive")
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if mutation_count <= 0:
        raise ValueError("mutation_count must be positive")

    rng = random.Random(seed)

    keys = list(range(record_count))
    rng.shuffle(keys)

    rid_by_key = {key: _rid(position) for position, key in enumerate(keys)}

    equality_keys = [rng.randrange(record_count) for _ in range(query_count)]

    range_width = max(1, record_count // 100)
    ranges: list[tuple[int, int]] = []

    for _ in range(query_count):
        lo = rng.randrange(record_count)
        hi = min(record_count - 1, lo + range_width)
        ranges.append((lo, hi))

    mutation_count = min(mutation_count, record_count)
    remove_keys = rng.sample(list(range(record_count)), mutation_count)

    insert_keys = list(
        range(
            record_count,
            record_count + mutation_count,
        )
    )

    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        workdir = Path(temporary_directory)

        for structure, factory in STRUCTURES.items():
            base_path = workdir / f"{structure}.db"

            results.append(
                _build_index(
                    structure,
                    factory,
                    keys,
                    base_path,
                )
            )

            results.append(
                _benchmark_equality(
                    structure,
                    factory,
                    base_path,
                    equality_keys,
                    records=record_count,
                )
            )

            results.append(
                _benchmark_range(
                    structure,
                    factory,
                    base_path,
                    ranges,
                    records=record_count,
                )
            )

            results.append(
                _benchmark_insert(
                    structure,
                    factory,
                    base_path,
                    workdir / f"{structure}-insert.db",
                    insert_keys,
                    records=record_count,
                )
            )

            results.append(
                _benchmark_remove(
                    structure,
                    factory,
                    base_path,
                    workdir / f"{structure}-remove.db",
                    remove_keys,
                    rid_by_key,
                    records=record_count,
                )
            )

    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'structure':<18}"
        f"{'operation':<18}"
        f"{'ms':>12}"
        f"{'ops/s':>14}"
        f"{'reads':>10}"
        f"{'writes':>10}"
        f"{'bytes':>12}"
        f"{'results':>10}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        operation = result.operation
        if not result.supported:
            operation += " (N/A)"

        print(
            f"{result.structure:<18}"
            f"{operation:<18}"
            f"{result.elapsed_ms:>12.3f}"
            f"{result.ops_per_second:>14.2f}"
            f"{result.disk_reads:>10}"
            f"{result.disk_writes:>10}"
            f"{result.size_bytes:>12}"
            f"{result.result_count:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare B+ and extendible-hash indexes.",
    )
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--mutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_benchmark(
        record_count=args.records,
        query_count=args.queries,
        mutation_count=args.mutations,
        seed=args.seed,
    )

    _print_results(results)

    path = write_results(
        "indexes",
        results,
    )

    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
