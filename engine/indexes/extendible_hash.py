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

        self._ensure_entry_fits(key, rid)

        while True:
            bucket_page_id = self._bucket_page_id(key)
            bucket = self._read_bucket(bucket_page_id)

            updated = codec.BucketPage(
                local_depth=bucket.local_depth,
                overflow_page_id=bucket.overflow_page_id,
                entries=(*bucket.entries, (key, rid)),
            )

            try:
                encoded = codec.encode_bucket(self._page_size, updated)
            except ValueError:
                if bucket.local_depth >= self._max_global_depth:
                    self._insert_into_overflow(bucket_page_id, key, rid)
                    return

                self._split_bucket(bucket_page_id, bucket)
                continue

            self._write_page(bucket_page_id, encoded)
            return

    def search(self, key: Key) -> list[RID]:
        self._ensure_open()

        bucket_page_id = self._bucket_page_id(key)
        result: list[RID] = []

        while bucket_page_id is not None:
            bucket = self._read_bucket(bucket_page_id)

            result.extend(rid for entry_key, rid in bucket.entries if entry_key == key)

            bucket_page_id = bucket.overflow_page_id

        return result

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

    def _directory_page_ids(self) -> list[int]:
        page_id: int | None = self._first_directory_page_id
        page_ids: list[int] = []

        while page_id is not None:
            page_ids.append(page_id)
            directory = codec.decode_directory(self._read_page(page_id))
            page_id = directory.next_page_id

        return page_ids

    def _write_directory(self, entries: tuple[int, ...]) -> None:
        capacity = codec.directory_capacity(self._page_size)

        chunks = [
            entries[position : position + capacity] for position in range(0, len(entries), capacity)
        ]

        if not chunks:
            raise ValueError("extendible-hash directory cannot be empty")

        page_ids = self._directory_page_ids()

        while len(page_ids) < len(chunks):
            page_ids.append(self._disk_manager.allocate_page())

        for position, chunk in enumerate(chunks):
            next_page_id = page_ids[position + 1] if position + 1 < len(chunks) else None

            directory = codec.DirectoryPage(
                next_page_id=next_page_id,
                bucket_page_ids=tuple(chunk),
            )

            self._write_page(
                page_ids[position],
                codec.encode_directory(self._page_size, directory),
            )

    def _bucket_page_id(self, key: Key) -> int:
        directory = self._read_directory()
        index = directory_index(key, self._global_depth)
        return directory[index]

    def _split_bucket(
        self,
        bucket_page_id: int,
        bucket: codec.BucketPage,
    ) -> None:
        directory = list(self._read_directory())
        old_local_depth = bucket.local_depth

        if old_local_depth >= self._max_global_depth:
            raise ValueError("maximum extendible-hash depth reached")

        if old_local_depth == self._global_depth:
            if self._global_depth >= self._max_global_depth:
                raise ValueError("maximum extendible-hash depth reached")

            directory.extend(directory)
            self._global_depth += 1

        new_local_depth = old_local_depth + 1
        split_bit = 1 << old_local_depth

        new_bucket_page_id = self._disk_manager.allocate_page()

        for index, page_id in enumerate(directory):
            if page_id == bucket_page_id and index & split_bit:
                directory[index] = new_bucket_page_id

        left_entries: list[tuple[Key, RID]] = []
        right_entries: list[tuple[Key, RID]] = []

        for entry_key, entry_rid in bucket.entries:
            index = directory_index(entry_key, self._global_depth)

            if directory[index] == bucket_page_id:
                left_entries.append((entry_key, entry_rid))
            else:
                right_entries.append((entry_key, entry_rid))

        left = codec.BucketPage(
            local_depth=new_local_depth,
            overflow_page_id=None,
            entries=tuple(left_entries),
        )
        right = codec.BucketPage(
            local_depth=new_local_depth,
            overflow_page_id=None,
            entries=tuple(right_entries),
        )

        self._write_page(
            bucket_page_id,
            codec.encode_bucket(self._page_size, left),
        )
        self._write_page(
            new_bucket_page_id,
            codec.encode_bucket(self._page_size, right),
        )

        self._write_directory(tuple(directory))
        self._write_metadata()

    def _insert_into_overflow(
        self,
        bucket_page_id: int,
        key: Key,
        rid: RID,
    ) -> None:
        current_page_id = bucket_page_id

        while True:
            bucket = self._read_bucket(current_page_id)

            updated = codec.BucketPage(
                local_depth=bucket.local_depth,
                overflow_page_id=bucket.overflow_page_id,
                entries=(*bucket.entries, (key, rid)),
            )

            try:
                encoded = codec.encode_bucket(self._page_size, updated)
            except ValueError:
                if bucket.overflow_page_id is not None:
                    current_page_id = bucket.overflow_page_id
                    continue

                overflow_page_id = self._disk_manager.allocate_page()

                linked = codec.BucketPage(
                    local_depth=bucket.local_depth,
                    overflow_page_id=overflow_page_id,
                    entries=bucket.entries,
                )
                self._write_page(
                    current_page_id,
                    codec.encode_bucket(self._page_size, linked),
                )

                overflow = codec.BucketPage(
                    local_depth=bucket.local_depth,
                    overflow_page_id=None,
                    entries=((key, rid),),
                )
                self._write_page(
                    overflow_page_id,
                    codec.encode_bucket(self._page_size, overflow),
                )
                return

            self._write_page(current_page_id, encoded)
            return

    def _ensure_entry_fits(self, key: Key, rid: RID) -> None:
        probe = codec.BucketPage(
            local_depth=0,
            overflow_page_id=None,
            entries=((key, rid),),
        )

        try:
            codec.encode_bucket(self._page_size, probe)
        except ValueError as exc:
            raise ValueError("index entry does not fit in a hash bucket page") from exc

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
