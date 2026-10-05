"""Persistent non-clustered B+ tree index."""

from bisect import bisect_left, bisect_right

from engine.common.rid import RID
from engine.indexes import _bplus_page as codec
from engine.indexes.base import Index, Key
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_METADATA_PAGE_ID = 0


class BPlusTree(Index):
    """Persistent non-clustered B+ tree backed by paged storage."""

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

        leaf_page_id = self._find_leaf_page_id(key, left_bias=False)
        leaf = self._read_leaf(leaf_page_id)

        entries = list(leaf.entries)
        entries.append((key, rid))
        entries.sort(key=lambda entry: (entry[0], entry[1].page_id, entry[1].slot))

        updated = codec.LeafNode(
            parent_page_id=leaf.parent_page_id,
            next_leaf_page_id=leaf.next_leaf_page_id,
            entries=tuple(entries),
        )

        try:
            page = codec.encode_leaf(self._page_size, updated)
        except ValueError:
            self._split_leaf(leaf_page_id, leaf, entries)
            return

        self._write_page(leaf_page_id, page)

    def search(self, key: Key) -> list[RID]:
        self._ensure_open()

        page_id = self._find_leaf_page_id(key, left_bias=True)
        result: list[RID] = []

        while page_id is not None:
            leaf = self._read_leaf(page_id)

            for entry_key, rid in leaf.entries:
                if entry_key < key:
                    continue
                if entry_key == key:
                    result.append(rid)
                    continue
                return result

            page_id = leaf.next_leaf_page_id

        return result

    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        self._ensure_open()

        if lo > hi:
            return []

        page_id = self._find_leaf_page_id(lo, left_bias=True)
        result: list[RID] = []

        while page_id is not None:
            leaf = self._read_leaf(page_id)

            for key, rid in leaf.entries:
                if key < lo:
                    continue
                if key > hi:
                    return result
                result.append(rid)

            page_id = leaf.next_leaf_page_id

        return result

    def remove(self, key: Key, rid: RID | None = None) -> int:
        self._ensure_open()

        page_id = self._find_leaf_page_id(key, left_bias=True)
        removed = 0

        while page_id is not None:
            leaf = self._read_leaf(page_id)
            remaining: list[tuple[Key, RID]] = []
            changed = False
            passed_key = False

            for entry_key, entry_rid in leaf.entries:
                if entry_key > key:
                    passed_key = True

                should_remove = entry_key == key and (
                    rid is None or (removed == 0 and entry_rid == rid)
                )

                if should_remove:
                    removed += 1
                    changed = True
                else:
                    remaining.append((entry_key, entry_rid))

            if changed:
                updated = codec.LeafNode(
                    parent_page_id=leaf.parent_page_id,
                    next_leaf_page_id=leaf.next_leaf_page_id,
                    entries=tuple(remaining),
                )
                self._write_page(
                    page_id,
                    codec.encode_leaf(self._page_size, updated),
                )

                if rid is not None:
                    return removed

            if passed_key:
                return removed

            page_id = leaf.next_leaf_page_id

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

        metadata = codec.decode_metadata(self._read_page(_METADATA_PAGE_ID))

        if metadata.root_page_id is None:
            raise ValueError("B+ metadata does not contain a root page")
        if metadata.first_leaf_page_id is None:
            raise ValueError("B+ metadata does not contain a first leaf page")

        self._root_page_id = metadata.root_page_id
        self._first_leaf_page_id = metadata.first_leaf_page_id

    def _write_metadata(self) -> None:
        self._write_page(
            _METADATA_PAGE_ID,
            codec.encode_metadata(
                self._page_size,
                root_page_id=self._root_page_id,
                first_leaf_page_id=self._first_leaf_page_id,
            ),
        )

    def _find_leaf_page_id(self, key: Key, *, left_bias: bool) -> int:
        page_id = self._root_page_id

        while True:
            page = self._read_page(page_id)
            node_type = codec.page_type(page)

            if node_type == codec.PAGE_TYPE_LEAF:
                return page_id

            if node_type != codec.PAGE_TYPE_INTERNAL:
                raise ValueError("invalid B+ node in search path")

            node = codec.decode_internal(page)

            if left_bias:
                child_index = bisect_left(node.keys, key)
            else:
                child_index = bisect_right(node.keys, key)

            page_id = node.children[child_index]

    def _split_leaf(
        self,
        leaf_page_id: int,
        leaf: codec.LeafNode,
        entries: list[tuple[Key, RID]],
    ) -> None:
        if leaf.parent_page_id is None:
            self._split_root_leaf(leaf_page_id, leaf, entries)
            return

        right_page_id = self._disk_manager.page_count

        left, right = self._choose_leaf_split(
            entries,
            parent_page_id=leaf.parent_page_id,
            right_page_id=right_page_id,
            old_next_leaf_page_id=leaf.next_leaf_page_id,
        )

        separator = right.entries[0][0]

        right_page_id = self._disk_manager.allocate_page()

        self._write_page(
            leaf_page_id,
            codec.encode_leaf(self._page_size, left),
        )
        self._write_page(
            right_page_id,
            codec.encode_leaf(self._page_size, right),
        )

        self._insert_into_parent(
            parent_page_id=leaf.parent_page_id,
            left_page_id=leaf_page_id,
            separator=separator,
            right_page_id=right_page_id,
        )

    def _split_root_leaf(
        self,
        leaf_page_id: int,
        leaf: codec.LeafNode,
        entries: list[tuple[Key, RID]],
    ) -> None:
        right_page_id = self._disk_manager.page_count
        new_root_page_id = right_page_id + 1

        left, right = self._choose_leaf_split(
            entries,
            parent_page_id=new_root_page_id,
            right_page_id=right_page_id,
            old_next_leaf_page_id=leaf.next_leaf_page_id,
        )

        separator = right.entries[0][0]

        new_root = codec.InternalNode(
            parent_page_id=None,
            keys=(separator,),
            children=(leaf_page_id, right_page_id),
        )

        allocated_right = self._disk_manager.allocate_page()
        allocated_root = self._disk_manager.allocate_page()

        if allocated_right != right_page_id or allocated_root != new_root_page_id:
            raise RuntimeError("unexpected B+ page allocation order")

        self._write_page(
            leaf_page_id,
            codec.encode_leaf(self._page_size, left),
        )
        self._write_page(
            right_page_id,
            codec.encode_leaf(self._page_size, right),
        )
        self._write_page(
            new_root_page_id,
            codec.encode_internal(self._page_size, new_root),
        )

        self._root_page_id = new_root_page_id
        self._write_metadata()

    def _insert_into_parent(
        self,
        *,
        parent_page_id: int,
        left_page_id: int,
        separator: Key,
        right_page_id: int,
    ) -> None:
        parent = self._read_internal(parent_page_id)

        try:
            child_position = parent.children.index(left_page_id)
        except ValueError as exc:
            raise ValueError("child is not referenced by its parent") from exc

        keys = list(parent.keys)
        children = list(parent.children)

        keys.insert(child_position, separator)
        children.insert(child_position + 1, right_page_id)

        updated = codec.InternalNode(
            parent_page_id=parent.parent_page_id,
            keys=tuple(keys),
            children=tuple(children),
        )

        try:
            encoded = codec.encode_internal(self._page_size, updated)
        except ValueError:
            self._split_internal(parent_page_id, updated)
            return

        self._write_page(parent_page_id, encoded)

    def _split_internal(
        self,
        page_id: int,
        node: codec.InternalNode,
    ) -> None:
        if node.parent_page_id is None:
            self._split_root_internal(page_id, node)
            return

        right_page_id = self._disk_manager.page_count

        left, promoted, right = self._choose_internal_split(
            node,
            left_parent_page_id=node.parent_page_id,
            right_parent_page_id=node.parent_page_id,
        )

        allocated_right = self._disk_manager.allocate_page()
        if allocated_right != right_page_id:
            raise RuntimeError("unexpected B+ page allocation order")

        self._write_page(
            page_id,
            codec.encode_internal(self._page_size, left),
        )
        self._write_page(
            right_page_id,
            codec.encode_internal(self._page_size, right),
        )

        self._update_children_parent(left.children, page_id)
        self._update_children_parent(right.children, right_page_id)

        self._insert_into_parent(
            parent_page_id=node.parent_page_id,
            left_page_id=page_id,
            separator=promoted,
            right_page_id=right_page_id,
        )

    def _split_root_internal(
        self,
        page_id: int,
        node: codec.InternalNode,
    ) -> None:
        right_page_id = self._disk_manager.page_count
        new_root_page_id = right_page_id + 1

        left, promoted, right = self._choose_internal_split(
            node,
            left_parent_page_id=new_root_page_id,
            right_parent_page_id=new_root_page_id,
        )

        new_root = codec.InternalNode(
            parent_page_id=None,
            keys=(promoted,),
            children=(page_id, right_page_id),
        )

        allocated_right = self._disk_manager.allocate_page()
        allocated_root = self._disk_manager.allocate_page()

        if allocated_right != right_page_id or allocated_root != new_root_page_id:
            raise RuntimeError("unexpected B+ page allocation order")

        self._write_page(
            page_id,
            codec.encode_internal(self._page_size, left),
        )
        self._write_page(
            right_page_id,
            codec.encode_internal(self._page_size, right),
        )

        self._update_children_parent(left.children, page_id)
        self._update_children_parent(right.children, right_page_id)

        self._write_page(
            new_root_page_id,
            codec.encode_internal(self._page_size, new_root),
        )

        self._root_page_id = new_root_page_id
        self._write_metadata()

    def _choose_leaf_split(
        self,
        entries: list[tuple[Key, RID]],
        *,
        parent_page_id: int,
        right_page_id: int,
        old_next_leaf_page_id: int | None,
    ) -> tuple[codec.LeafNode, codec.LeafNode]:
        candidates = sorted(
            range(1, len(entries)),
            key=lambda position: abs(len(entries) - 2 * position),
        )

        for position in candidates:
            left = codec.LeafNode(
                parent_page_id=parent_page_id,
                next_leaf_page_id=right_page_id,
                entries=tuple(entries[:position]),
            )
            right = codec.LeafNode(
                parent_page_id=parent_page_id,
                next_leaf_page_id=old_next_leaf_page_id,
                entries=tuple(entries[position:]),
            )

            try:
                codec.encode_leaf(self._page_size, left)
                codec.encode_leaf(self._page_size, right)
            except ValueError:
                continue

            return left, right

        raise ValueError("B+ leaf entries cannot be split into valid pages")

    def _choose_internal_split(
        self,
        node: codec.InternalNode,
        *,
        left_parent_page_id: int,
        right_parent_page_id: int,
    ) -> tuple[codec.InternalNode, Key, codec.InternalNode]:
        candidates = sorted(
            range(len(node.keys)),
            key=lambda position: abs(len(node.keys) - 1 - 2 * position),
        )

        for position in candidates:
            promoted = node.keys[position]

            left = codec.InternalNode(
                parent_page_id=left_parent_page_id,
                keys=node.keys[:position],
                children=node.children[: position + 1],
            )
            right = codec.InternalNode(
                parent_page_id=right_parent_page_id,
                keys=node.keys[position + 1 :],
                children=node.children[position + 1 :],
            )

            if not left.keys or not right.keys:
                continue

            try:
                codec.encode_internal(self._page_size, left)
                codec.encode_internal(self._page_size, right)
            except ValueError:
                continue

            return left, promoted, right

        raise ValueError("B+ internal node cannot be split into valid pages")

    def _update_children_parent(
        self,
        children: tuple[int, ...],
        parent_page_id: int,
    ) -> None:
        for child_page_id in children:
            self._set_parent_page_id(child_page_id, parent_page_id)

    def _set_parent_page_id(
        self,
        page_id: int,
        parent_page_id: int,
    ) -> None:
        page = self._read_page(page_id)
        node_type = codec.page_type(page)

        if node_type == codec.PAGE_TYPE_LEAF:
            leaf = codec.decode_leaf(page)
            updated = codec.LeafNode(
                parent_page_id=parent_page_id,
                next_leaf_page_id=leaf.next_leaf_page_id,
                entries=leaf.entries,
            )
            encoded = codec.encode_leaf(self._page_size, updated)

        elif node_type == codec.PAGE_TYPE_INTERNAL:
            internal = codec.decode_internal(page)
            updated = codec.InternalNode(
                parent_page_id=parent_page_id,
                keys=internal.keys,
                children=internal.children,
            )
            encoded = codec.encode_internal(self._page_size, updated)

        else:
            raise ValueError("metadata page cannot be an internal child")

        self._write_page(page_id, encoded)

    def _read_leaf(self, page_id: int) -> codec.LeafNode:
        page = self._read_page(page_id)

        if codec.page_type(page) != codec.PAGE_TYPE_LEAF:
            raise ValueError("expected B+ leaf page")

        return codec.decode_leaf(page)

    def _read_internal(self, page_id: int) -> codec.InternalNode:
        page = self._read_page(page_id)

        if codec.page_type(page) != codec.PAGE_TYPE_INTERNAL:
            raise ValueError("expected B+ internal page")

        return codec.decode_internal(page)

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
