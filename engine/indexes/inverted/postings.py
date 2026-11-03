"""Versioned binary postings: uint64 count followed by (uint64 doc, uint64 tf)."""

import struct
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

_HEADER = struct.Struct("<4sI")
_MAGIC = b"SPST"
_VERSION = 1
_COUNT = struct.Struct("<Q")
_PAIR = struct.Struct("<QQ")


def _validate(postings: Sequence[tuple[int, int]]) -> None:
    previous = -1
    for doc_id, tf in postings:
        if type(doc_id) is not int or not previous < doc_id < 2**64:
            raise ValueError("posting document IDs must be ordered unique uint64 values")
        if type(tf) is not int or not 0 < tf < 2**64:
            raise ValueError("posting frequencies must be positive uint64 values")
        previous = doc_id


class PostingsFile:
    """Append-only binary lists; offsets and lengths belong to the term dictionary."""

    def __init__(self, file: BinaryIO, *, writable: bool) -> None:
        self._file = file
        self._writable = writable

    @classmethod
    def create(cls, path: str | Path) -> "PostingsFile":
        file = Path(path).open("x+b")
        try:
            file.write(_HEADER.pack(_MAGIC, _VERSION))
            return cls(file, writable=True)
        except BaseException:
            file.close()
            raise

    @classmethod
    def load(cls, path: str | Path) -> "PostingsFile":
        file = Path(path).open("rb")
        try:
            if file.read(_HEADER.size) != _HEADER.pack(_MAGIC, _VERSION):
                raise ValueError("invalid postings magic or version")
            return cls(file, writable=False)
        except BaseException:
            file.close()
            raise

    def append(self, postings: Sequence[tuple[int, int]]) -> tuple[int, int]:
        if not self._writable or self._file.closed:
            raise ValueError("postings file is not writable")
        _validate(postings)
        self._file.seek(0, 2)
        offset = self._file.tell()
        self._file.write(_COUNT.pack(len(postings)))
        for doc_id, tf in postings:
            self._file.write(_PAIR.pack(doc_id, tf))
        return offset, _COUNT.size + len(postings) * _PAIR.size

    def read(self, offset: int, length: int, *, df: int) -> list[tuple[int, int]]:
        if self._file.closed:
            raise ValueError("postings file is closed")
        if (
            type(offset) is not int or offset < _HEADER.size
            or type(length) is not int or type(df) is not int or df < 0
            or length != _COUNT.size + df * _PAIR.size
        ):
            raise ValueError("invalid postings offset, length or document frequency")
        self._file.seek(0, 2)
        if offset + length > self._file.tell():
            raise ValueError("postings list exceeds file bounds")
        self._file.seek(offset)
        if _COUNT.unpack(self._file.read(_COUNT.size))[0] != df:
            raise ValueError("postings count differs from dictionary frequency")
        data = self._file.read(df * _PAIR.size)
        if len(data) != df * _PAIR.size:
            raise ValueError("truncated postings list")
        postings = list(_PAIR.iter_unpack(data))
        _validate(postings)
        return postings

    def close(self) -> None:
        self._file.close()
