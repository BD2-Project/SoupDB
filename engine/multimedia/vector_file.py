"""Formato vectores.bin: puente entre la extracción multimedia y los índices vectoriales.

Little-endian. Cabecera ``<4sIII`` (magic, version, count, dimension) seguida de
``count`` registros ``(id: int64, path_len: uint32, path: UTF-8, vector: K float32)``.
"""

import random
import struct
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import BinaryIO, NamedTuple

MAGIC = b"VEC1"
VERSION = 1

_HEADER = struct.Struct("<4sIII")
_RECORD_HEAD = struct.Struct("<qI")


class VectorRecord(NamedTuple):
    id: int
    path: str
    vector: Sequence[float]


def write_vectors(
    path: str | Path, records: Iterable[VectorRecord], *, dimension: int, count: int
) -> None:
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    vector_layout = struct.Struct(f"<{dimension}f")

    try:
        with open(path, "wb") as f:
            f.write(_HEADER.pack(MAGIC, VERSION, count, dimension))
            written = 0
            for record_id, record_path, vector in records:
                if written == count:
                    raise ValueError(f"more than {count} records given")
                if len(vector) != dimension:
                    raise ValueError(
                        f"record {record_id}: vector has {len(vector)} values, expected {dimension}"
                    )
                encoded = record_path.encode("utf-8")
                f.write(_RECORD_HEAD.pack(record_id, len(encoded)))
                f.write(encoded)
                f.write(vector_layout.pack(*vector))
                written += 1
            if written != count:
                raise ValueError(f"expected {count} records, got {written}")
    except BaseException:
        # Un archivo a medias tendría una cabecera que miente; mejor no dejarlo.
        Path(path).unlink(missing_ok=True)
        raise


def iter_vectors(path: str | Path) -> Iterator[VectorRecord]:
    with open(path, "rb") as f:
        magic, version, count, dimension = _HEADER.unpack(_read_exact(f, _HEADER.size))
        if magic != MAGIC:
            raise ValueError(f"not a vectores.bin file (magic {magic!r})")
        if version != VERSION:
            raise ValueError(f"unsupported vectores.bin version {version}")
        if dimension == 0:
            raise ValueError("invalid dimension 0")
        vector_layout = struct.Struct(f"<{dimension}f")

        for _ in range(count):
            record_id, path_len = _RECORD_HEAD.unpack(_read_exact(f, _RECORD_HEAD.size))
            record_path = _read_exact(f, path_len).decode("utf-8")
            vector = vector_layout.unpack(_read_exact(f, vector_layout.size))
            yield VectorRecord(record_id, record_path, vector)


def random_vectors(count: int, dimension: int, seed: int) -> Iterator[VectorRecord]:
    rng = random.Random(seed)
    for i in range(count):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        yield VectorRecord(i, f"synthetic/{i}", vector)


def _read_exact(f: BinaryIO, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"truncated vectores.bin: expected {size} bytes, got {len(data)}")
    return data
