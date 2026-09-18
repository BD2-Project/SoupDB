"""Benchmark external sorting, GROUP BY, and hash JOIN."""

import argparse
import random
import time

from benchmarks.harness import BenchmarkResult, write_results
from engine.algorithms.external_hash import (
    external_hash_group_by,
    external_hash_join,
)
from engine.algorithms.external_sort import external_sort
from engine.common.record import Record


def _number_record(value: int) -> Record:
    return Record(data=str(value).encode())


def _number(record: Record) -> int:
    return int(record.data)


def _compare_numbers(left: Record, right: Record) -> int:
    return (_number(left) > _number(right)) - (_number(left) < _number(right))


def _pair_record(key: int, value: int) -> Record:
    return Record(data=f"{key}:{value}".encode())


def _pair_key(record: Record) -> int:
    return int(record.data.split(b":", 1)[0])


def _count(current: int | None, _record: Record) -> int:
    return 1 if current is None else current + 1


def _rate(operations: int, elapsed_seconds: float) -> float:
    if elapsed_seconds == 0:
        return 0.0
    return operations / elapsed_seconds


def _benchmark_sort(
    records: list[Record],
    memory_limit_bytes: int,
) -> BenchmarkResult:
    started = time.perf_counter()

    result = list(
        external_sort(
            records,
            _compare_numbers,
            memory_limit_bytes=memory_limit_bytes,
        )
    )

    elapsed = time.perf_counter() - started

    values = [_number(record) for record in result]
    if values != sorted(values):
        raise RuntimeError("external sort produced an invalid result")

    return BenchmarkResult(
        structure=f"external_sort_{memory_limit_bytes}B",
        operation="order_by",
        records=len(records),
        operations=len(records),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(records), elapsed),
        disk_reads=0,
        disk_writes=0,
        size_bytes=0,
        result_count=len(result),
    )


def _benchmark_group_by(
    records: list[Record],
    memory_limit_bytes: int,
) -> BenchmarkResult:
    started = time.perf_counter()

    result = dict(
        external_hash_group_by(
            records,
            _pair_key,
            _count,
            memory_limit_bytes=memory_limit_bytes,
        )
    )

    elapsed = time.perf_counter() - started

    if sum(result.values()) != len(records):
        raise RuntimeError("external GROUP BY produced an invalid result")

    return BenchmarkResult(
        structure=f"external_hash_{memory_limit_bytes}B",
        operation="group_by",
        records=len(records),
        operations=len(records),
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(len(records), elapsed),
        disk_reads=0,
        disk_writes=0,
        size_bytes=0,
        result_count=len(result),
    )


def _benchmark_join(
    left: list[Record],
    right: list[Record],
    memory_limit_bytes: int,
) -> BenchmarkResult:
    started = time.perf_counter()

    result = list(
        external_hash_join(
            left,
            right,
            _pair_key,
            _pair_key,
            memory_limit_bytes=memory_limit_bytes,
        )
    )

    elapsed = time.perf_counter() - started

    for left_record, right_record in result:
        if _pair_key(left_record) != _pair_key(right_record):
            raise RuntimeError("external JOIN produced an invalid result")

    operations = len(left) + len(right)

    return BenchmarkResult(
        structure=f"external_hash_{memory_limit_bytes}B",
        operation="join",
        records=operations,
        operations=operations,
        elapsed_ms=elapsed * 1000,
        ops_per_second=_rate(operations, elapsed),
        disk_reads=0,
        disk_writes=0,
        size_bytes=0,
        result_count=len(result),
    )


def run_benchmark(
    *,
    record_count: int,
    memory_limits: list[int],
    seed: int,
) -> list[BenchmarkResult]:
    if record_count <= 0:
        raise ValueError("record_count must be positive")

    if not memory_limits:
        raise ValueError("at least one memory limit is required")

    if any(limit <= 0 for limit in memory_limits):
        raise ValueError("memory limits must be positive")

    rng = random.Random(seed)

    values = list(range(record_count))
    rng.shuffle(values)

    sort_records = [_number_record(value) for value in values]

    group_count = max(1, record_count // 100)
    group_records = [_pair_record(value % group_count, value) for value in values]

    join_key_count = max(1, record_count // 10)

    left_records = [_pair_record(value % join_key_count, value) for value in range(record_count)]

    right_records = [
        _pair_record(value % join_key_count, value) for value in range(record_count // 2)
    ]

    results: list[BenchmarkResult] = []

    for memory_limit in memory_limits:
        results.append(
            _benchmark_sort(
                sort_records,
                memory_limit,
            )
        )

        results.append(
            _benchmark_group_by(
                group_records,
                memory_limit,
            )
        )

        results.append(
            _benchmark_join(
                left_records,
                right_records,
                memory_limit,
            )
        )

    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'configuration':<28}"
        f"{'operation':<14}"
        f"{'records':>10}"
        f"{'ms':>12}"
        f"{'ops/s':>14}"
        f"{'results':>12}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.structure:<28}"
            f"{result.operation:<14}"
            f"{result.records:>10}"
            f"{result.elapsed_ms:>12.3f}"
            f"{result.ops_per_second:>14.2f}"
            f"{result.result_count:>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SoupDB external algorithms.",
    )
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument(
        "--memory",
        type=int,
        nargs="+",
        default=[4096, 16384, 65536],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_benchmark(
        record_count=args.records,
        memory_limits=args.memory,
        seed=args.seed,
    )

    _print_results(results)

    path = write_results(
        "external_algorithms",
        results,
    )

    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
