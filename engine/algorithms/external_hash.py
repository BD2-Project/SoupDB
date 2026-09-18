"""External hashing for GROUP BY and hash JOIN."""

import hashlib
import struct
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import BinaryIO, TypeVar

from engine.common.record import Record
from engine.indexes._key_codec import encode_key
from engine.indexes.base import Key

Aggregate = TypeVar("Aggregate")

_LENGTH_FORMAT = "<I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FORMAT)


def external_hash_group_by(
    records: Iterable[Record],
    key_fn: Callable[[Record], Key],
    aggregate_step: Callable[[Aggregate | None, Record], Aggregate],
    *,
    memory_limit_bytes: int,
    partition_count: int = 8,
    max_partition_depth: int = 4,
    temp_dir: str | Path | None = None,
) -> Iterator[tuple[Key, Aggregate]]:
    """Group records using disk-backed hash partitioning."""
    _validate_parameters(
        memory_limit_bytes,
        partition_count,
        max_partition_depth,
    )

    with tempfile.TemporaryDirectory(dir=temp_dir) as working_directory:
        workdir = Path(working_directory)

        partitions = _partition_records(
            records,
            key_fn,
            workdir=workdir,
            prefix="group",
            partition_count=partition_count,
            level=0,
        )

        for partition in partitions:
            yield from _group_partition(
                partition,
                key_fn,
                aggregate_step,
                workdir=workdir,
                memory_limit_bytes=memory_limit_bytes,
                partition_count=partition_count,
                level=1,
                max_partition_depth=max_partition_depth,
            )


def external_hash_join(
    left_records: Iterable[Record],
    right_records: Iterable[Record],
    left_key_fn: Callable[[Record], Key],
    right_key_fn: Callable[[Record], Key],
    *,
    memory_limit_bytes: int,
    partition_count: int = 8,
    max_partition_depth: int = 4,
    temp_dir: str | Path | None = None,
) -> Iterator[tuple[Record, Record]]:
    """Perform an equality join using external hash partitioning."""
    _validate_parameters(
        memory_limit_bytes,
        partition_count,
        max_partition_depth,
    )

    with tempfile.TemporaryDirectory(dir=temp_dir) as working_directory:
        workdir = Path(working_directory)

        left_partitions = _partition_records(
            left_records,
            left_key_fn,
            workdir=workdir,
            prefix="join-left",
            partition_count=partition_count,
            level=0,
        )
        right_partitions = _partition_records(
            right_records,
            right_key_fn,
            workdir=workdir,
            prefix="join-right",
            partition_count=partition_count,
            level=0,
        )

        for left_path, right_path in zip(
            left_partitions,
            right_partitions,
            strict=True,
        ):
            yield from _join_partition(
                left_path,
                right_path,
                left_key_fn,
                right_key_fn,
                workdir=workdir,
                memory_limit_bytes=memory_limit_bytes,
                partition_count=partition_count,
                level=1,
                max_partition_depth=max_partition_depth,
            )


def _group_partition(
    path: Path,
    key_fn: Callable[[Record], Key],
    aggregate_step: Callable[[Aggregate | None, Record], Aggregate],
    *,
    workdir: Path,
    memory_limit_bytes: int,
    partition_count: int,
    level: int,
    max_partition_depth: int,
) -> Iterator[tuple[Key, Aggregate]]:
    if path.stat().st_size > memory_limit_bytes and level <= max_partition_depth:
        children = _partition_records(
            _read_records(path),
            key_fn,
            workdir=workdir,
            prefix=f"group-{level}-{path.stem}",
            partition_count=partition_count,
            level=level,
        )

        for child in children:
            yield from _group_partition(
                child,
                key_fn,
                aggregate_step,
                workdir=workdir,
                memory_limit_bytes=memory_limit_bytes,
                partition_count=partition_count,
                level=level + 1,
                max_partition_depth=max_partition_depth,
            )
        return

    groups: dict[Key, Aggregate] = {}

    for record in _read_records(path):
        key = key_fn(record)
        groups[key] = aggregate_step(groups.get(key), record)

    yield from groups.items()


def _join_partition(
    left_path: Path,
    right_path: Path,
    left_key_fn: Callable[[Record], Key],
    right_key_fn: Callable[[Record], Key],
    *,
    workdir: Path,
    memory_limit_bytes: int,
    partition_count: int,
    level: int,
    max_partition_depth: int,
) -> Iterator[tuple[Record, Record]]:
    if left_path.stat().st_size == 0 or right_path.stat().st_size == 0:
        return

    smaller_size = min(
        left_path.stat().st_size,
        right_path.stat().st_size,
    )

    if smaller_size <= memory_limit_bytes:
        yield from _in_memory_join(
            left_path,
            right_path,
            left_key_fn,
            right_key_fn,
        )
        return

    if level <= max_partition_depth:
        left_children = _partition_records(
            _read_records(left_path),
            left_key_fn,
            workdir=workdir,
            prefix=f"left-{level}-{left_path.stem}",
            partition_count=partition_count,
            level=level,
        )
        right_children = _partition_records(
            _read_records(right_path),
            right_key_fn,
            workdir=workdir,
            prefix=f"right-{level}-{right_path.stem}",
            partition_count=partition_count,
            level=level,
        )

        for child_left, child_right in zip(
            left_children,
            right_children,
            strict=True,
        ):
            yield from _join_partition(
                child_left,
                child_right,
                left_key_fn,
                right_key_fn,
                workdir=workdir,
                memory_limit_bytes=memory_limit_bytes,
                partition_count=partition_count,
                level=level + 1,
                max_partition_depth=max_partition_depth,
            )
        return

    yield from _block_hash_join(
        left_path,
        right_path,
        left_key_fn,
        right_key_fn,
        memory_limit_bytes=memory_limit_bytes,
    )


def _in_memory_join(
    left_path: Path,
    right_path: Path,
    left_key_fn: Callable[[Record], Key],
    right_key_fn: Callable[[Record], Key],
) -> Iterator[tuple[Record, Record]]:
    build_left = left_path.stat().st_size <= right_path.stat().st_size

    if build_left:
        build_path = left_path
        probe_path = right_path
        build_key_fn = left_key_fn
        probe_key_fn = right_key_fn
    else:
        build_path = right_path
        probe_path = left_path
        build_key_fn = right_key_fn
        probe_key_fn = left_key_fn

    table: dict[Key, list[Record]] = {}

    for record in _read_records(build_path):
        table.setdefault(build_key_fn(record), []).append(record)

    for probe_record in _read_records(probe_path):
        matches = table.get(probe_key_fn(probe_record), ())

        for build_record in matches:
            if build_left:
                yield build_record, probe_record
            else:
                yield probe_record, build_record


def _block_hash_join(
    left_path: Path,
    right_path: Path,
    left_key_fn: Callable[[Record], Key],
    right_key_fn: Callable[[Record], Key],
    *,
    memory_limit_bytes: int,
) -> Iterator[tuple[Record, Record]]:
    build_left = left_path.stat().st_size <= right_path.stat().st_size

    if build_left:
        build_path = left_path
        probe_path = right_path
        build_key_fn = left_key_fn
        probe_key_fn = right_key_fn
    else:
        build_path = right_path
        probe_path = left_path
        build_key_fn = right_key_fn
        probe_key_fn = left_key_fn

    for chunk in _record_chunks(
        build_path,
        memory_limit_bytes=memory_limit_bytes,
    ):
        table: dict[Key, list[Record]] = {}

        for record in chunk:
            table.setdefault(build_key_fn(record), []).append(record)

        for probe_record in _read_records(probe_path):
            matches = table.get(probe_key_fn(probe_record), ())

            for build_record in matches:
                if build_left:
                    yield build_record, probe_record
                else:
                    yield probe_record, build_record


def _partition_records(
    records: Iterable[Record],
    key_fn: Callable[[Record], Key],
    *,
    workdir: Path,
    prefix: str,
    partition_count: int,
    level: int,
) -> list[Path]:
    paths = [
        workdir / f"{prefix}-level-{level}-partition-{index}.bin"
        for index in range(partition_count)
    ]

    files = [path.open("wb") for path in paths]

    try:
        for record in records:
            partition = _partition_index(
                key_fn(record),
                partition_count=partition_count,
                level=level,
            )
            _write_record(files[partition], record)
    finally:
        for file in files:
            file.close()

    return paths


def _partition_index(
    key: Key,
    *,
    partition_count: int,
    level: int,
) -> int:
    encoded = encode_key(key)
    salt = level.to_bytes(4, byteorder="little", signed=False)
    digest = hashlib.blake2b(
        encoded + salt,
        digest_size=8,
    ).digest()

    return (
        int.from_bytes(
            digest,
            byteorder="little",
            signed=False,
        )
        % partition_count
    )


def _record_chunks(
    path: Path,
    *,
    memory_limit_bytes: int,
) -> Iterator[list[Record]]:
    chunk: list[Record] = []
    used_bytes = 0

    for record in _read_records(path):
        size = _serialized_size(record)

        if chunk and used_bytes + size > memory_limit_bytes:
            yield chunk
            chunk = []
            used_bytes = 0

        chunk.append(record)
        used_bytes += size

        if used_bytes >= memory_limit_bytes:
            yield chunk
            chunk = []
            used_bytes = 0

    if chunk:
        yield chunk


def _write_record(file: BinaryIO, record: Record) -> None:
    data = record.data

    if len(data) > 0xFFFFFFFF:
        raise ValueError("record is too large for external-hash format")

    file.write(struct.pack(_LENGTH_FORMAT, len(data)))
    file.write(data)


def _read_records(path: Path) -> Iterator[Record]:
    with path.open("rb") as file:
        while True:
            length_data = file.read(_LENGTH_SIZE)

            if not length_data:
                return

            if len(length_data) != _LENGTH_SIZE:
                raise ValueError("truncated external-hash record length")

            length = struct.unpack(_LENGTH_FORMAT, length_data)[0]
            data = file.read(length)

            if len(data) != length:
                raise ValueError("truncated external-hash record")

            yield Record(data=data)


def _serialized_size(record: Record) -> int:
    return _LENGTH_SIZE + len(record.data)


def _validate_parameters(
    memory_limit_bytes: int,
    partition_count: int,
    max_partition_depth: int,
) -> None:
    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    if partition_count < 2:
        raise ValueError("partition_count must be at least 2")
    if max_partition_depth < 0:
        raise ValueError("max_partition_depth must be non-negative")
