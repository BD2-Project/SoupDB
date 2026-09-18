"""Memory-bounded external sorting using temporary runs and k-way merge."""

import heapq
import struct
import tempfile
from collections.abc import Callable, Iterable, Iterator
from functools import cmp_to_key
from pathlib import Path
from typing import BinaryIO

from engine.common.record import Record

Comparator = Callable[[Record, Record], int]

_LENGTH_FORMAT = "<I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FORMAT)


def external_sort(
    records: Iterable[Record],
    comparator: Comparator,
    *,
    memory_limit_bytes: int,
    fan_in: int = 8,
    temp_dir: str | Path | None = None,
) -> Iterator[Record]:
    """Sort records externally using bounded runs and k-way merge.

    ``memory_limit_bytes`` bounds the payload accumulated for each initial
    in-memory run. ``fan_in`` limits the number of runs opened in each merge.
    """

    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    if fan_in < 2:
        raise ValueError("fan_in must be at least 2")

    order_key = cmp_to_key(comparator)

    with tempfile.TemporaryDirectory(dir=temp_dir) as working_directory:
        workdir = Path(working_directory)

        runs = _create_initial_runs(
            records,
            workdir=workdir,
            order_key=order_key,
            memory_limit_bytes=memory_limit_bytes,
        )

        if not runs:
            return

        pass_number = 0

        while len(runs) > 1:
            merged_runs: list[Path] = []

            for group_number, start in enumerate(range(0, len(runs), fan_in)):
                group = runs[start : start + fan_in]

                if len(group) == 1:
                    merged_runs.append(group[0])
                    continue

                output = workdir / f"pass-{pass_number}-run-{group_number}.bin"
                _merge_runs(
                    group,
                    output,
                    order_key=order_key,
                )
                merged_runs.append(output)

            runs = merged_runs
            pass_number += 1

        yield from _read_run(runs[0])


def _create_initial_runs(
    records: Iterable[Record],
    *,
    workdir: Path,
    order_key,
    memory_limit_bytes: int,
) -> list[Path]:
    runs: list[Path] = []
    chunk: list[Record] = []
    used_bytes = 0

    for record in records:
        record_size = _serialized_size(record)

        if chunk and used_bytes + record_size > memory_limit_bytes:
            runs.append(
                _write_sorted_run(
                    chunk,
                    workdir=workdir,
                    run_number=len(runs),
                    order_key=order_key,
                )
            )
            chunk = []
            used_bytes = 0

        chunk.append(record)
        used_bytes += record_size

        if used_bytes >= memory_limit_bytes:
            runs.append(
                _write_sorted_run(
                    chunk,
                    workdir=workdir,
                    run_number=len(runs),
                    order_key=order_key,
                )
            )
            chunk = []
            used_bytes = 0

    if chunk:
        runs.append(
            _write_sorted_run(
                chunk,
                workdir=workdir,
                run_number=len(runs),
                order_key=order_key,
            )
        )

    return runs


def _write_sorted_run(
    records: list[Record],
    *,
    workdir: Path,
    run_number: int,
    order_key,
) -> Path:
    records.sort(key=order_key)

    path = workdir / f"initial-run-{run_number}.bin"

    with path.open("wb") as file:
        for record in records:
            _write_record(file, record)

    return path


def _merge_runs(
    inputs: list[Path],
    output: Path,
    *,
    order_key,
) -> None:
    files: list[BinaryIO] = []

    try:
        files = [path.open("rb") for path in inputs]
        heap: list[tuple[object, int, int, Record]] = []
        sequence = 0

        for run_index, file in enumerate(files):
            record = _read_record(file)

            if record is not None:
                heapq.heappush(
                    heap,
                    (
                        order_key(record),
                        sequence,
                        run_index,
                        record,
                    ),
                )
                sequence += 1

        with output.open("wb") as output_file:
            while heap:
                _key, _sequence, run_index, record = heapq.heappop(heap)
                _write_record(output_file, record)

                next_record = _read_record(files[run_index])

                if next_record is not None:
                    heapq.heappush(
                        heap,
                        (
                            order_key(next_record),
                            sequence,
                            run_index,
                            next_record,
                        ),
                    )
                    sequence += 1

    finally:
        for file in files:
            file.close()


def _read_run(path: Path) -> Iterator[Record]:
    with path.open("rb") as file:
        while True:
            record = _read_record(file)

            if record is None:
                return

            yield record


def _write_record(file: BinaryIO, record: Record) -> None:
    data = record.data

    if len(data) > 0xFFFFFFFF:
        raise ValueError("record is too large for external-sort run format")

    file.write(struct.pack(_LENGTH_FORMAT, len(data)))
    file.write(data)


def _read_record(file: BinaryIO) -> Record | None:
    length_data = file.read(_LENGTH_SIZE)

    if not length_data:
        return None

    if len(length_data) != _LENGTH_SIZE:
        raise ValueError("truncated external-sort record length")

    length = struct.unpack(_LENGTH_FORMAT, length_data)[0]
    data = file.read(length)

    if len(data) != length:
        raise ValueError("truncated external-sort record")

    return Record(data=data)


def _serialized_size(record: Record) -> int:
    return _LENGTH_SIZE + len(record.data)
