"""Persistent non-clustered B+ tree index."""

from engine.common.rid import RID
from engine.indexes import _bplus_page as codec
from engine.indexes.base import Index, Key
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_METADATA_PAGE_ID = 0


class BPlusTree(Index):
    """Persistent B+ tree backed by DiskManager and BufferManager.

    This initial implementation supports a single leaf page. Node splitting is
    added separately.
    """

    def __init__(
        self,
        disk_manager: DiskManager,
        buffer_manager: BufferManager,
    ) -> None:
        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager
        self._page_size = disk_manager.page_size
        self._closed = False

        if disk_manager.page_count == 0:
            self._initialize()
        else:
            self._load_metadata()

    @property
    def supports_range(self) -> bool:
        return True

    def insert(self, key: Key, rid: RID) -> None:
        self._ensure_open()

        leaf = self._read_root_leaf()
        entries = list(leaf.entries)
        entries.append((key, rid))
        entries.sort(key=lambda entry: (entry[0], entry[1].page_id, entry[1].slot))

        updated = codec.LeafNode(
            parent_page_id=leaf.parent_page_id,
            next_leaf_page_id=leaf.next_leaf_page_id,
            entries=tuple(entries),
        )

        page = codec.encode_leaf(self._page_size, updated)
        self._write_page(self._root_page_id, page)

    def search(self, key: Key) -> list[RID]:
        self._ensure_open()

        leaf = self._read_root_leaf()
        return [rid for entry_key, rid in leaf.entries if entry_key == key]

    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        self._ensure_open()

        leaf = self._read_root_leaf()
        return [rid for key, rid in leaf.entries if lo <= key <= hi]

    def remove(self, key: Key, rid: RID | None = None) -> int:
        self._ensure_open()

        leaf = self._read_root_leaf()
        entries = list(leaf.entries)

        if rid is None:
            remaining = [
                (entry_key, entry_rid) for entry_key, entry_rid in entries if entry_key != key
            ]
            removed = len(entries) - len(remaining)
        else:
            remaining = list(entries)
            removed = 0

            for position, (entry_key, entry_rid) in enumerate(entries):
                if entry_key == key and entry_rid == rid:
                    del remaining[position]
                    removed = 1
                    break

        if removed == 0:
            return 0

        updated = codec.LeafNode(
            parent_page_id=leaf.parent_page_id,
            next_leaf_page_id=leaf.next_leaf_page_id,
            entries=tuple(remaining),
        )
        self._write_page(
            self._root_page_id,
            codec.encode_leaf(self._page_size, updated),
        )
        return removed

    def close(self) -> None:
        if self._closed:
            return

        self._buffer_manager.flush_all()
        self._closed = True

    def _initialize(self) -> None:
        metadata_page_id = self._disk_manager.allocate_page()
        if metadata_page_id != _METADATA_PAGE_ID:
            raise ValueError("B+ metadata page must be page 0")

        root_page_id = self._disk_manager.allocate_page()

        root = codec.LeafNode(
            parent_page_id=None,
            next_leaf_page_id=None,
            entries=(),
        )
        self._write_page(
            root_page_id,
            codec.encode_leaf(self._page_size, root),
        )

        self._root_page_id = root_page_id
        self._first_leaf_page_id = root_page_id
        self._write_metadata()

    def _load_metadata(self) -> None:
        if self._disk_manager.page_count < 2:
            raise ValueError("incomplete B+ index file")

        page = self._read_page(_METADATA_PAGE_ID)
        metadata = codec.decode_metadata(page)

        if metadata.root_page_id is None:
            raise ValueError("B+ metadata does not contain a root page")
        if metadata.first_leaf_page_id is None:
            raise ValueError("B+ metadata does not contain a first leaf page")

        self._root_page_id = metadata.root_page_id
        self._first_leaf_page_id = metadata.first_leaf_page_id

    def _write_metadata(self) -> None:
        page = codec.encode_metadata(
            self._page_size,
            root_page_id=self._root_page_id,
            first_leaf_page_id=self._first_leaf_page_id,
        )
        self._write_page(_METADATA_PAGE_ID, page)

    def _read_root_leaf(self) -> codec.LeafNode:
        page = self._read_page(self._root_page_id)

        if codec.page_type(page) != codec.PAGE_TYPE_LEAF:
            raise ValueError("B+ root is not a leaf in single-leaf mode")

        return codec.decode_leaf(page)

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
            raise RuntimeError("B+ tree is closed")
