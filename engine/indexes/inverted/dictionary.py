"""ExtendibleHash term lookup with auxiliary RIDs into a binary metadata sidecar.

Auxiliary RID ordinal = page_id * 65536 + slot; these are never document RIDs.
The sidecar contains term UTF-8 bytes and uint64 df/offset/length. Only its record
offsets are kept in memory; term lookup always uses ExtendibleHash.search.
"""

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from engine.common.rid import RID
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_HEADER = struct.Struct("<4sI")
_RECORD = struct.Struct("<IQQQ")
_MAGIC = b"SDCT"
_VERSION = 1


@dataclass(frozen=True)
class DictionaryEntry:
    df: int
    offset: int
    length: int


class TermDictionary:
    """Append-only term metadata; close flushes the owned hash buffer pool."""

    def __init__(self, file: BinaryIO, disk: DiskManager, index: ExtendibleHash,
                 *, page_size: int, writable: bool) -> None:
        self._file = file
        self._disk = disk
        self._index = index
        self._page_size = page_size
        self._writable = writable
        self._offsets: list[int] = []

    @classmethod
    def _open(cls, directory: str | Path, *, page_size: int, create: bool) -> "TermDictionary":
        if type(page_size) is not int or page_size < 128:
            raise ValueError("dictionary page_size must be an integer >=128")
        directory = Path(directory)
        meta_path, hash_path = directory / "terms.meta", directory / "terms.hash"
        if create:
            if meta_path.exists() or hash_path.exists():
                raise FileExistsError("dictionary already exists")
            directory.mkdir(parents=True, exist_ok=True)
        elif not meta_path.is_file() or not hash_path.is_file():
            raise FileNotFoundError("dictionary artifacts are missing")
        elif hash_path.stat().st_size == 0:
            raise ValueError("empty dictionary hash file")
        file = meta_path.open("x+b" if create else "rb")
        disk = None
        try:
            if create:
                file.write(_HEADER.pack(_MAGIC, _VERSION))
            elif file.read(_HEADER.size) != _HEADER.pack(_MAGIC, _VERSION):
                raise ValueError("invalid dictionary magic or version")
            disk = DiskManager(hash_path, page_size=page_size)
            index = ExtendibleHash(disk, BufferManager(disk, capacity=64))
            result = cls(file, disk, index, page_size=page_size, writable=create)
            if not create:
                result._scan()
            return result
        except BaseException:
            file.close()
            if disk is not None:
                disk.close()
            raise

    @classmethod
    def create(cls, directory: str | Path, *, page_size: int = 4096) -> "TermDictionary":
        return cls._open(directory, page_size=page_size, create=True)

    @classmethod
    def load(cls, directory: str | Path, *, page_size: int = 4096) -> "TermDictionary":
        return cls._open(directory, page_size=page_size, create=False)

    def _check(self) -> None:
        if self._file.closed:
            raise ValueError("term dictionary is closed")

    def _read(self, offset: int) -> tuple[str, DictionaryEntry]:
        self._file.seek(offset)
        header = self._file.read(_RECORD.size)
        if len(header) != _RECORD.size:
            raise ValueError("truncated dictionary record")
        size, df, location, length = _RECORD.unpack(header)
        if not 0 < size <= self._page_size - 18:
            raise ValueError("invalid dictionary term size")
        raw = self._file.read(size)
        if len(raw) != size:
            raise ValueError("truncated dictionary term")
        if df == 0 or location < 8 or length != 8 + df * 16:
            raise ValueError("invalid dictionary postings metadata")
        return raw.decode("utf-8"), DictionaryEntry(df, location, length)

    def _scan(self) -> None:
        self._file.seek(0, 2)
        size = self._file.tell()
        offset = _HEADER.size
        while offset < size:
            term, _entry = self._read(offset)
            ordinal = len(self._offsets)
            expected = RID(ordinal // 65536, ordinal % 65536)
            if self._index.search(term) != [expected]:
                raise ValueError("dictionary hash and sidecar disagree")
            self._offsets.append(offset)
            offset = self._file.tell()

    def add(self, term: str, entry: DictionaryEntry) -> None:
        self._check()
        if not self._writable:
            raise ValueError("dictionary is read-only")
        if not isinstance(term, str) or not term:
            raise ValueError("dictionary terms must be non-empty strings")
        encoded = term.encode("utf-8")
        # Hash entry overhead: bucket(9), key length(2), RID(6), key tag(1).
        if len(encoded) > self._page_size - 18:
            raise ValueError("term does not fit in an ExtendibleHash bucket")
        if (any(type(value) is not int or not 0 <= value < 2**64
                for value in (entry.df, entry.offset, entry.length))
                or entry.df == 0 or entry.offset < 8 or entry.length != 8 + entry.df * 16):
            raise ValueError("invalid dictionary entry")
        if self._index.search(term):
            raise ValueError("duplicate dictionary term")
        ordinal = len(self._offsets)
        if ordinal >= 2**48:
            raise ValueError("dictionary auxiliary RID capacity exceeded")
        self._file.seek(0, 2)
        offset = self._file.tell()
        self._file.write(_RECORD.pack(len(encoded), entry.df, entry.offset, entry.length))
        self._file.write(encoded)
        self._index.insert(term, RID(ordinal // 65536, ordinal % 65536))
        self._offsets.append(offset)

    def find(self, term: str) -> DictionaryEntry | None:
        self._check()
        references = self._index.search(term)
        if not references:
            return None
        if len(references) != 1:
            raise ValueError("dictionary term has multiple auxiliary references")
        rid = references[0]
        ordinal = rid.page_id * 65536 + rid.slot
        if ordinal >= len(self._offsets):
            raise ValueError("dictionary reference exceeds sidecar bounds")
        stored, entry = self._read(self._offsets[ordinal])
        if stored != term:
            raise ValueError("dictionary reference points to another term")
        return entry

    def items(self) -> Iterator[tuple[str, DictionaryEntry]]:
        self._check()
        for offset in self._offsets:
            yield self._read(offset)

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            self._index.close()
        finally:
            self._file.close()
            self._disk.close()
