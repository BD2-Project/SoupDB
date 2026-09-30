"""Compare physical index strategies: clustered B+, unclustered B+, extendible hash."""

import argparse
import random
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from benchmarks.harness import BenchmarkResult, write_results
from engine.common.record import Record
from engine.common.rid import RID
from engine.indexes.bplus_tree import BPlusTree
from engine.indexes.clustered_bplus import ClusteredBPlusTree
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.heap_file import HeapFile

PAGE_SIZE = 4096
BUFFER_CAPACITY = 32
SEED = 42
PAYLOAD_SIZE = 64

IndexFactory = Callable[[DiskManager, BufferManager], BPlusTree | ExtendibleHash]


def _record(key: int) -> Record:
    prefix = f"{key}:".encode()
    return Record(data=prefix + b"x" * PAYLOAD_SIZE)


def _key(record: Record) -> int:
    return int(record.data.split(b":", 1)[0])


def _rate(operations: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return operations / elapsed_seconds


def _size(index_path: Path, data_path: Path) -> int:
    return index_path.stat().st_size + data_path.stat().st_size


def _io(index_dm: DiskManager, data_dm: DiskManager) -> tuple[int, int]:
    return (index_dm.reads + data_dm.reads, index_dm.writes + data_dm.writes)


def _open_bplus(dm: DiskManager, bm: BufferManager) -> BPlusTree:
    return BPlusTree(dm, bm)


def _open_hash(dm: DiskManager, bm: BufferManager) -> ExtendibleHash:
    return ExtendibleHash(dm, bm)


def _verify_ascending(keys: list[int]) -> None:
    for previous, current in zip(keys, keys[1:], strict=False):
        if current <= previous:
            raise AssertionError(f"ordered_fetch keys not ascending: {previous} then {current}")


def _build_unclustered_bplus(keys: list[int], index_path: Path, data_path: Path) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)

    started = time.perf_counter()

    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = BPlusTree(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    for key in keys:
        rid = data_file.insert(_record(key))
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


def _build_clustered_bplus(keys: list[int], index_path: Path, data_path: Path) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)

    started = time.perf_counter()

    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = ClusteredBPlusTree(index_dm, index_bm, data_dm, data_bm, _key)

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


def _build_hash(keys: list[int], index_path: Path, data_path: Path) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)

    started = time.perf_counter()

    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    index = ExtendibleHash(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    for key in keys:
        rid = data_file.insert(_record(key))
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
    index_factory: IndexFactory,
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

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

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

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

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

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = ClusteredBPlusTree(index_dm, index_bm, data_dm, data_bm, _key)

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

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = ClusteredBPlusTree(index_dm, index_bm, data_dm, data_bm, _key)

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


def _ordered_unclustered(
    structure: str,
    index_factory: IndexFactory,
    index_path: Path,
    data_path: Path,
    *,
    records: int,
) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    rids = index.range_search(0, records - 1)
    keys: list[int] = []
    for rid in rids:
        record = data_file.fetch(rid)
        if record is not None:
            keys.append(_key(record))

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    _verify_ascending(keys)
    result_count = len(keys)

    result = BenchmarkResult(
        structure=structure,
        operation="ordered_fetch",
        records=records,
        operations=result_count,
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(result_count, elapsed),
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        size_bytes=_size(index_path, data_path),
        result_count=result_count,
    )

    index.close()
    index_dm.close()
    data_dm.close()
    return result


def _ordered_clustered(index_path: Path, data_path: Path, *, records: int) -> BenchmarkResult:
    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = ClusteredBPlusTree(index_dm, index_bm, data_dm, data_bm, _key)
    records_out = index.range_records(0, records - 1)

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    _verify_ascending([_key(record) for record in records_out])
    result_count = len(records_out)

    result = BenchmarkResult(
        structure="bplus_clustered",
        operation="ordered_fetch",
        records=records,
        operations=result_count,
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(result_count, elapsed),
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        size_bytes=_size(index_path, data_path),
        result_count=result_count,
    )

    index.close()
    index_dm.close()
    data_dm.close()
    return result


def _ordered_hash_unsupported(
    index_path: Path, data_path: Path, *, records: int
) -> BenchmarkResult:
    return BenchmarkResult(
        structure="extendible_hash",
        operation="ordered_fetch",
        records=records,
        operations=0,
        elapsed_ms=0.0,
        ops_per_second=0.0,
        disk_reads=0,
        disk_writes=0,
        size_bytes=_size(index_path, data_path),
        result_count=0,
        supported=False,
    )


def _mutation_insert_unclustered(
    structure: str,
    index_factory: IndexFactory,
    base_index_path: Path,
    base_data_path: Path,
    working_index_path: Path,
    working_data_path: Path,
    new_keys: list[int],
    *,
    records: int,
) -> BenchmarkResult:
    shutil.copyfile(base_index_path, working_index_path)
    shutil.copyfile(base_data_path, working_data_path)

    index_dm = DiskManager(working_index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(working_data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    for key in new_keys:
        rid = data_file.insert(_record(key))
        index.insert(key, rid)

    data_bm.flush_all()
    index_bm.flush_all()

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure=structure,
        operation="mutation_insert",
        records=records,
        operations=len(new_keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(new_keys), elapsed),
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        size_bytes=_size(working_index_path, working_data_path),
        result_count=len(new_keys),
    )

    index.close()
    index_dm.close()
    data_dm.close()
    return result


def _mutation_insert_clustered(
    base_index_path: Path,
    base_data_path: Path,
    working_index_path: Path,
    working_data_path: Path,
    new_keys: list[int],
    *,
    records: int,
) -> BenchmarkResult:
    shutil.copyfile(base_index_path, working_index_path)
    shutil.copyfile(base_data_path, working_data_path)

    index_dm = DiskManager(working_index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(working_data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = ClusteredBPlusTree(index_dm, index_bm, data_dm, data_bm, _key)

    for key in new_keys:
        index.insert_record(_record(key))

    data_bm.flush_all()
    index_bm.flush_all()

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure="bplus_clustered",
        operation="mutation_insert",
        records=records,
        operations=len(new_keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(new_keys), elapsed),
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        size_bytes=_size(working_index_path, working_data_path),
        result_count=len(new_keys),
    )

    index.close()
    index_dm.close()
    data_dm.close()
    return result


def _mutation_remove_unclustered(
    structure: str,
    index_factory: IndexFactory,
    base_index_path: Path,
    base_data_path: Path,
    working_index_path: Path,
    working_data_path: Path,
    remove_keys: list[int],
    *,
    records: int,
) -> BenchmarkResult:
    shutil.copyfile(base_index_path, working_index_path)
    shutil.copyfile(base_data_path, working_data_path)

    index_dm = DiskManager(working_index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(working_data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = index_factory(index_dm, index_bm)
    data_file = HeapFile(data_dm, data_bm)

    removed = 0
    for key in remove_keys:
        rids: list[RID] = index.search(key)
        if not rids:
            continue
        rid = rids[0]
        if data_file.remove(rid):
            index.remove(key, rid)
            removed += 1

    data_bm.flush_all()
    index_bm.flush_all()

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure=structure,
        operation="mutation_remove",
        records=records,
        operations=len(remove_keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(remove_keys), elapsed),
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        size_bytes=_size(working_index_path, working_data_path),
        result_count=removed,
    )

    index.close()
    index_dm.close()
    data_dm.close()
    return result


def _mutation_remove_clustered(
    base_index_path: Path,
    base_data_path: Path,
    working_index_path: Path,
    working_data_path: Path,
    remove_keys: list[int],
    *,
    records: int,
) -> BenchmarkResult:
    shutil.copyfile(base_index_path, working_index_path)
    shutil.copyfile(base_data_path, working_data_path)

    index_dm = DiskManager(working_index_path, page_size=PAGE_SIZE)
    data_dm = DiskManager(working_data_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=BUFFER_CAPACITY)
    data_bm = BufferManager(data_dm, capacity=BUFFER_CAPACITY)

    reads_before, writes_before = _io(index_dm, data_dm)
    started = time.perf_counter()

    index = ClusteredBPlusTree(index_dm, index_bm, data_dm, data_bm, _key)

    removed = 0
    for key in remove_keys:
        rids: list[RID] = index.search(key)
        if not rids:
            continue
        rid = rids[0]
        if index.remove_record(key, rid):
            removed += 1

    data_bm.flush_all()
    index_bm.flush_all()

    elapsed = time.perf_counter() - started
    reads_after, writes_after = _io(index_dm, data_dm)

    result = BenchmarkResult(
        structure="bplus_clustered",
        operation="mutation_remove",
        records=records,
        operations=len(remove_keys),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(remove_keys), elapsed),
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        size_bytes=_size(working_index_path, working_data_path),
        result_count=removed,
    )

    index.close()
    index_dm.close()
    data_dm.close()
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

    equality_keys = [rng.randrange(record_count) for _ in range(query_count)]

    range_width = max(1, record_count // 100)
    ranges: list[tuple[int, int]] = []
    for _ in range(query_count):
        lo = rng.randrange(record_count)
        hi = min(record_count - 1, lo + range_width)
        ranges.append((lo, hi))

    mutation_count = min(mutation_count, record_count)
    remove_keys = rng.sample(list(range(record_count)), mutation_count)
    new_keys = list(range(record_count, record_count + mutation_count))

    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)

        unclustered_index = workdir / "bplus-unclustered-index.db"
        unclustered_data = workdir / "bplus-unclustered-data.db"
        clustered_index = workdir / "bplus-clustered-index.db"
        clustered_data = workdir / "bplus-clustered-data.db"
        hash_index = workdir / "hash-index.db"
        hash_data = workdir / "hash-data.db"

        results.append(_build_unclustered_bplus(keys, unclustered_index, unclustered_data))
        results.append(_build_clustered_bplus(keys, clustered_index, clustered_data))
        results.append(_build_hash(keys, hash_index, hash_data))

        results.extend(
            _query_unclustered(
                "bplus_unclustered",
                _open_bplus,
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
                clustered_index, clustered_data, equality_keys, ranges, records=record_count
            )
        )
        results.extend(
            _query_unclustered(
                "extendible_hash",
                _open_hash,
                hash_index,
                hash_data,
                equality_keys,
                ranges,
                records=record_count,
                supports_range=False,
            )
        )

        results.append(
            _ordered_unclustered(
                "bplus_unclustered",
                _open_bplus,
                unclustered_index,
                unclustered_data,
                records=record_count,
            )
        )
        results.append(_ordered_clustered(clustered_index, clustered_data, records=record_count))
        results.append(_ordered_hash_unsupported(hash_index, hash_data, records=record_count))

        results.append(
            _mutation_insert_unclustered(
                "bplus_unclustered",
                _open_bplus,
                unclustered_index,
                unclustered_data,
                workdir / "bplus-unclustered-index-ins.db",
                workdir / "bplus-unclustered-data-ins.db",
                new_keys,
                records=record_count,
            )
        )
        results.append(
            _mutation_insert_clustered(
                clustered_index,
                clustered_data,
                workdir / "bplus-clustered-index-ins.db",
                workdir / "bplus-clustered-data-ins.db",
                new_keys,
                records=record_count,
            )
        )
        results.append(
            _mutation_insert_unclustered(
                "extendible_hash",
                _open_hash,
                hash_index,
                hash_data,
                workdir / "hash-index-ins.db",
                workdir / "hash-data-ins.db",
                new_keys,
                records=record_count,
            )
        )

        results.append(
            _mutation_remove_unclustered(
                "bplus_unclustered",
                _open_bplus,
                unclustered_index,
                unclustered_data,
                workdir / "bplus-unclustered-index-rm.db",
                workdir / "bplus-unclustered-data-rm.db",
                remove_keys,
                records=record_count,
            )
        )
        results.append(
            _mutation_remove_clustered(
                clustered_index,
                clustered_data,
                workdir / "bplus-clustered-index-rm.db",
                workdir / "bplus-clustered-data-rm.db",
                remove_keys,
                records=record_count,
            )
        )
        results.append(
            _mutation_remove_unclustered(
                "extendible_hash",
                _open_hash,
                hash_index,
                hash_data,
                workdir / "hash-index-rm.db",
                workdir / "hash-data-rm.db",
                remove_keys,
                records=record_count,
            )
        )

    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'structure':<20} "
        f"{'operation':<18} "
        f"{'records':>10} "
        f"{'operations':>12} "
        f"{'ms':>14} "
        f"{'ops/s':>14} "
        f"{'reads':>10} "
        f"{'writes':>10} "
        f"{'bytes':>12} "
        f"{'results':>10} "
        f"{'supported':>10}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.structure:<20} "
            f"{result.operation:<18} "
            f"{result.records:>10} "
            f"{result.operations:>12} "
            f"{result.elapsed_ms:>14.3f} "
            f"{result.ops_per_second:>14.2f} "
            f"{result.disk_reads:>10} "
            f"{result.disk_writes:>10} "
            f"{result.size_bytes:>12} "
            f"{result.result_count:>10} "
            f"{str(result.supported):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare physical SoupDB index strategies.")
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--mutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    results = run_benchmark(
        record_count=args.records,
        query_count=args.queries,
        mutation_count=args.mutations,
        seed=args.seed,
    )

    _print_results(results)

    path = write_results("physical_indexes", results)

    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
