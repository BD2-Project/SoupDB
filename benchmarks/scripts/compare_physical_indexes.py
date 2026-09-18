"""Compare physical index strategies including record retrieval."""

import argparse
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from benchmarks.harness import BenchmarkResult, write_results
from engine.common.record import Record
from engine.indexes.bplus_tree import BPlusTree
from engine.indexes.clustered_bplus import ClusteredBPlusTree
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.heap_file import HeapFile

PAGE_SIZE = 4096
BUFFER_CAPACITY = 32
PAYLOAD_SIZE = 64


def _record(key: int) -> Record:
    prefix = f"{key}:".encode()
    return Record(data=prefix + b"x" * PAYLOAD_SIZE)


def _key(record: Record) -> int:
    return int(record.data.split(b":", 1)[0])


def _rate(operations: int, elapsed_seconds: float) -> float:
    if elapsed_seconds == 0:
        return 0.0
    return operations / elapsed_seconds


def _size(index_path: Path, data_path: Path) -> int:
    return index_path.stat().st_size + data_path.stat().st_size


def _io(
    index_dm: DiskManager,
    data_dm: DiskManager,
) -> tuple[int, int]:
    return (
        index_dm.reads + data_dm.reads,
        index_dm.writes + data_dm.writes,
    )


def _build_unclustered_bplus(
    keys: list[int],
    index_path: Path,
    data_path: Path,
) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)

    started = time.perf_counter()

    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = BPlusTree(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    for key in keys:
        record = _record(key)
        rid = data_file.insert(record)
        index.insert(key, rid)

    data_bm.flush_all()
    index.close()

    elapsed = time.perf_counter() - started
    reads, writes = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure="bplus_unclustered",
        operation="build",
        records=len(keys),
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=reads,
        disk_writes=writes,
        size_bytes=_size(index_path, data_path),
        result_count=len(keys),
    )

    index_dm.close()
    data_dm.close()
    return result


def _build_clustered_bplus(
    keys: list[int],
    index_path: Path,
    data_path: Path,
) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)

    started = time.perf_counter()

    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = ClusteredBPlusTree(
        index_dm,
        index_bm,
        data_dm,
        data_bm,
        _key,
    )

    for key in keys:
        index.insert_record(_record(key))

    index.close()

    elapsed = time.perf_counter() - started
    reads, writes = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure="bplus_clustered",
        operation="build",
        records=len(keys),
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=reads,
        disk_writes=writes,
        size_bytes=_size(index_path, data_path),
        result_count=len(keys),
    )

    index_dm.close()
    data_dm.close()
    return result


def _build_hash(
    keys: list[int],
    index_path: Path,
    data_path: Path,
) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)

    started = time.perf_counter()

    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = ExtendibleHash(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    for key in keys:
        record = _record(key)
        rid = data_file.insert(record)
        index.insert(key, rid)

    data_bm.flush_all()
    index.close()

    elapsed = time.perf_counter() - started
    reads, writes = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure="extendible_hash",
        operation="build",
        records=len(keys),
        operations=len(keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(keys), elapsed),
        disk_reads=reads,
        disk_writes=writes,
        size_bytes=_size(index_path, data_path),
        result_count=len(keys),
    )

    index_dm.close()
    data_dm.close()
    return result


def _query_unclustered(
    structure: str,
    index_factory: Callable[[DiskManager, BufferManager], object],
    index_path: Path,
    data_path: Path,
    equality_keys: list[int],
    ranges: list[tuple[int, int]],
    *,
    records: int,
    supports_range: bool,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    reads_before, writes_before = _io(index_dm, data_dm)

    started = time.perf_counter()
    result_count = 0

    for key in equality_keys:
        for rid in index.search(key):
            if data_file.fetch(rid) is not None:
                result_count += 1

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    results.append(
        BenchmarkResult(
            structure=structure,
            operation="equality_fetch",
            records=records,
            operations=len(equality_keys),
            elapsed_ms=elapsed * 1000,
            ops_per_second=_rate(len(equality_keys), elapsed),
            disk_reads=reads_after - reads_before,
            disk_writes=writes_after - writes_before,
            size_bytes=_size(index_path, data_path),
            result_count=result_count,
        )
    )

    index.close()
    index_dm.close()
    data_dm.close()

    if not supports_range:
        results.append(
            BenchmarkResult(
                structure=structure,
                operation="range_fetch",
                records=records,
                operations=len(ranges),
                elapsed_ms=0.0,
                ops_per_second=0.0,
                disk_reads=0,
                disk_writes=0,
                size_bytes=_size(index_path, data_path),
                result_count=0,
                supported=False,
            )
        )
        return results

    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    reads_before, writes_before = _io(index_dm, data_dm)

    started = time.perf_counter()
    result_count = 0

    for lo, hi in ranges:
        for rid in index.range_search(lo, hi):
            if data_file.fetch(rid) is not None:
                result_count += 1

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    results.append(
        BenchmarkResult(
            structure=structure,
            operation="range_fetch",
            records=records,
            operations=len(ranges),
            elapsed_ms=elapsed * 1000,
            ops_per_second=_rate(len(ranges), elapsed),
            disk_reads=reads_after - reads_before,
            disk_writes=writes_after - writes_before,
            size_bytes=_size(index_path, data_path),
            result_count=result_count,
        )
    )

    index.close()
    index_dm.close()
    data_dm.close()

    return results


def _query_clustered(
    index_path: Path,
    data_path: Path,
    equality_keys: list[int],
    ranges: list[tuple[int, int]],
    *,
    records: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = ClusteredBPlusTree(
        index_dm,
        index_bm,
        data_dm,
        data_bm,
        _key,
    )

    reads_before, writes_before = _io(index_dm, data_dm)

    started = time.perf_counter()
    result_count = 0

    for key in equality_keys:
        result_count += len(index.search_records(key))

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    results.append(
        BenchmarkResult(
            structure="bplus_clustered",
            operation="equality_fetch",
            records=records,
            operations=len(equality_keys),
            elapsed_ms=elapsed * 1000,
            ops_per_second=_rate(len(equality_keys), elapsed),
            disk_reads=reads_after - reads_before,
            disk_writes=writes_after - writes_before,
            size_bytes=_size(index_path, data_path),
            result_count=result_count,
        )
    )

    index.close()
    index_dm.close()
    data_dm.close()

    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = ClusteredBPlusTree(
        index_dm,
        index_bm,
        data_dm,
        data_bm,
        _key,
    )

    reads_before, writes_before = _io(index_dm, data_dm)

    started = time.perf_counter()
    result_count = 0

    for lo, hi in ranges:
        result_count += len(index.range_records(lo, hi))

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    results.append(
        BenchmarkResult(
            structure="bplus_clustered",
            operation="range_fetch",
            records=records,
            operations=len(ranges),
            elapsed_ms=elapsed * 1000,
            ops_per_second=_rate(len(ranges), elapsed),
            disk_reads=reads_after - reads_before,
            disk_writes=writes_after - writes_before,
            size_bytes=_size(index_path, data_path),
            result_count=result_count,
        )
    )

    index.close()
    index_dm.close()
    data_dm.close()

    return results


def run_benchmark(
    *,
    record_count: int,
    query_count: int,
    seed: int,
) -> list[BenchmarkResult]:
    if record_count <= 0:
        raise ValueError("record_count must be positive")
    if query_count <= 0:
        raise ValueError("query_count must be positive")

    rng = random.Random(seed)

    keys = list(range(record_count))
    rng.shuffle(keys)

    equality_keys = [rng.randrange(record_count) for _ in range(query_count)]

    range_width = max(1, record_count // 100)

    ranges = []
    for _ in range(query_count):
        lo = rng.randrange(record_count)
        hi = min(record_count - 1, lo + range_width)
        ranges.append((lo, hi))

    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)

        unclustered_index = workdir / "bplus-unclustered-index.db"
        unclustered_data = workdir / "bplus-unclustered-data.db"

        clustered_index = workdir / "bplus-clustered-index.db"
        clustered_data = workdir / "bplus-clustered-data.db"

        hash_index = workdir / "hash-index.db"
        hash_data = workdir / "hash-data.db"

        results.append(
            _build_unclustered_bplus(
                keys,
                unclustered_index,
                unclustered_data,
            )
        )

        results.append(
            _build_clustered_bplus(
                keys,
                clustered_index,
                clustered_data,
            )
        )

        results.append(
            _build_hash(
                keys,
                hash_index,
                hash_data,
            )
        )

        results.extend(
            _query_unclustered(
                "bplus_unclustered",
                lambda dm, bm: BPlusTree(dm, bm),
                unclustered_index,
                unclustered_data,
                equality_keys,
                ranges,
                records=record_count,
                supports_range=True,
            )
        )

        results.extend(
            _query_clustered(
                clustered_index,
                clustered_data,
                equality_keys,
                ranges,
                records=record_count,
            )
        )

        results.extend(
            _query_unclustered(
                "extendible_hash",
                lambda dm, bm: ExtendibleHash(dm, bm),
                hash_index,
                hash_data,
                equality_keys,
                ranges,
                records=record_count,
                supports_range=False,
            )
        )

    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'strategy':<20}"
        f"{'operation':<20}"
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
            f"{result.structure:<20}"
            f"{operation:<20}"
            f"{result.elapsed_ms:>12.3f}"
            f"{result.ops_per_second:>14.2f}"
            f"{result.disk_reads:>10}"
            f"{result.disk_writes:>10}"
            f"{result.size_bytes:>12}"
            f"{result.result_count:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare physical SoupDB index strategies.",
    )
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_benchmark(
        record_count=args.records,
        query_count=args.queries,
        seed=args.seed,
    )

    _print_results(results)

    path = write_results(
        "physical_indexes",
        results,
    )

    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
