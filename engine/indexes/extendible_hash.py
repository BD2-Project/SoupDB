"""Persistent extendible-hash index."""

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes import _extendible_hash_page as codec
from engine.indexes._stable_hash import directory_index
from engine.indexes.base import Index, Key
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_METADATA_PAGE_ID = 0


class ExtendibleHash(Index):
    """Persistent equality index based on extendible hashing."""

    def __init__(
        self,
        disk_manager: DiskManager,
        buffer_manager: BufferManager,
        *,
        max_global_depth: int = 16,
    ) -> None:
        if max_global_depth < 0 or max_global_depth > 64:
            raise ValueError("max_global_depth must be between 0 and 64")

        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager
        self._page_size = disk_manager.page_size
        self._closed = False

        if disk_manager.page_count == 0:
            self._max_global_depth = max_global_depth
            self._initialize()
        else:
            self._load_metadata()

    @property
    def supports_range(self) -> bool:
        return False

    def insert(self, key: Key, rid: RID) -> None:
        self._ensure_open()

        bucket_page_id = self._bucket_page_id(key)
        bucket = self._read_bucket(bucket_page_id)

        entries = list(bucket.entries)
        entries.append((key, rid))

        updated = codec.BucketPage(
            local_depth=bucket.local_depth,
            overflow_page_id=bucket.overflow_page_id,
            entries=tuple(entries),
        )

        try:
            encoded = codec.encode_bucket(self._page_size, updated)
        except ValueError as exc:
            raise ValueError("extendible-hash bucket split required") from exc

        self._write_page(bucket_page_id, encoded)

    def search(self, key: Key) -> list[RID]:
        self._ensure_open()

        bucket_page_id = self._bucket_page_id(key)
        bucket = self._read_bucket(bucket_page_id)

        result: list[RID] = []

        while True:
            result.extend(rid for entry_key, rid in bucket.entries if entry_key == key)

            if bucket.overflow_page_id is None:
                return result

            bucket = self._read_bucket(bucket.overflow_page_id)

    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        self._ensure_open()
        raise UnsupportedOperation("ExtendibleHash does not support range searches")

    def remove(self, key: Key, rid: RID | None = None) -> int:
        self._ensure_open()

        bucket_page_id = self._bucket_page_id(key)
        removed = 0

        while bucket_page_id is not None:
            bucket = self._read_bucket(bucket_page_id)
            remaining: list[tuple[Key, RID]] = []

            for entry_key, entry_rid in bucket.entries:
                should_remove = entry_key == key and (
                    rid is None or (removed == 0 and entry_rid == rid)
                )

                if should_remove:
                    removed += 1
                else:
                    remaining.append((entry_key, entry_rid))

            if len(remaining) != len(bucket.entries):
                updated = codec.BucketPage(
                    local_depth=bucket.local_depth,
                    overflow_page_id=bucket.overflow_page_id,
                    entries=tuple(remaining),
                )
                self._write_page(
                    bucket_page_id,
                    codec.encode_bucket(self._page_size, updated),
                )

                if rid is not None:
                    return removed

            bucket_page_id = bucket.overflow_page_id

        return removed

    def close(self) -> None:
        if self._closed:
            return

        self._buffer_manager.flush_all()
        self._closed = True

    def _initialize(self) -> None:
        metadata_page_id = self._disk_manager.allocate_page()
        directory_page_id = self._disk_manager.allocate_page()
        bucket_page_id = self._disk_manager.allocate_page()

        if metadata_page_id != _METADATA_PAGE_ID:
            raise ValueError("extendible-hash metadata page must be page 0")

        self._global_depth = 0
        self._first_directory_page_id = directory_page_id

        bucket = codec.BucketPage(
            local_depth=0,
            overflow_page_id=None,
            entries=(),
        )
        self._write_page(
            bucket_page_id,
            codec.encode_bucket(self._page_size, bucket),
        )

        directory = codec.DirectoryPage(
            next_page_id=None,
            bucket_page_ids=(bucket_page_id,),
        )
        self._write_page(
            directory_page_id,
            codec.encode_directory(self._page_size, directory),
        )

        self._write_metadata()

    def _load_metadata(self) -> None:
        if self._disk_manager.page_count < 3:
            raise ValueError("incomplete extendible-hash index file")

        metadata = codec.decode_metadata(self._read_page(_METADATA_PAGE_ID))

        self._global_depth = metadata.global_depth
        self._first_directory_page_id = metadata.first_directory_page_id
        self._max_global_depth = metadata.max_global_depth

        expected_entries = 1 << self._global_depth
        actual_entries = len(self._read_directory())

        if actual_entries != expected_entries:
            raise ValueError("extendible-hash directory size does not match global depth")

    def _write_metadata(self) -> None:
        page = codec.encode_metadata(
            self._page_size,
            global_depth=self._global_depth,
            first_directory_page_id=self._first_directory_page_id,
            max_global_depth=self._max_global_depth,
        )
        self._write_page(_METADATA_PAGE_ID, page)

    def _read_directory(self) -> tuple[int, ...]:
        page_id: int | None = self._first_directory_page_id
        entries: list[int] = []

        while page_id is not None:
            directory = codec.decode_directory(self._read_page(page_id))
            entries.extend(directory.bucket_page_ids)
            page_id = directory.next_page_id

        return tuple(entries)

    def _bucket_page_id(self, key: Key) -> int:
        directory = self._read_directory()
        index = directory_index(key, self._global_depth)
        return directory[index]

    def _read_bucket(self, page_id: int) -> codec.BucketPage:
        page = self._read_page(page_id)

        if codec.page_type(page) != codec.PAGE_TYPE_BUCKET:
            raise ValueError("expected extendible-hash bucket page")

        return codec.decode_bucket(page)

    def _read_page(self, page_id: int) -> bytes:
        frame = self._buffer_manager.pin(page_id)

        try:
            return bytes(frame)
        finally:
            self._buffer_manager.unpin(page_id, dirty=False)

    def _write_page(self, page_id: int, data: bytes) -> None:
        frame = self._buffer_manager.pin(page_id)

        try:
            frame[:] = data
        finally:
            self._buffer_manager.unpin(page_id, dirty=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("extendible hash is closed")
