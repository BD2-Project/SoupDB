"""Persistencia paginada del R-Tree sobre DiskManager/BufferManager.

El árbol sigue viviendo en memoria como ``RTreeNode``; esta capa lo vuelca a
una página por nodo (``_page_codec``) y lo reconstruye al cargar.

Página 0 = metadata (big-endian, rellena con ceros hasta ``page_size``):

.. code-block::

    magic "RTM1" (4B) | version u8 | reserved u8 = 0 | order u16 | root_page_id i64

``root_page_id = -1`` indica árbol vacío.

Limitación: no hay free-list; las páginas de nodos podados por borrados quedan
huérfanas y no se reutilizan.
"""

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engine.indexes.rtree import _page_codec
from engine.indexes.rtree._page_codec import DecodedRTreePage
from engine.indexes.rtree.node import RTreeNode, refresh_mbr
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

if TYPE_CHECKING:
    from engine.indexes.rtree.rtree import RTree

METADATA_PAGE_ID = 0
EMPTY_ROOT_PAGE_ID = -1

_MAGIC = b"RTM1"
_VERSION = 1
_METADATA = struct.Struct(">4sBBHq")
_MIN_ORDER = 2


@dataclass(frozen=True)
class RTreeMetadata:
    order: int
    root_page_id: int


def encode_metadata(page_size: int, metadata: RTreeMetadata) -> bytes:
    page = bytearray(page_size)
    _METADATA.pack_into(page, 0, _MAGIC, _VERSION, 0, metadata.order, metadata.root_page_id)
    return bytes(page)


def decode_metadata(page: bytes) -> RTreeMetadata:
    if len(page) < _METADATA.size:
        raise ValueError("truncated R-Tree metadata page")
    magic, version, reserved, order, root_page_id = _METADATA.unpack_from(page, 0)
    if magic != _MAGIC:
        raise ValueError(f"invalid R-Tree metadata magic {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported R-Tree metadata version {version}")
    if reserved != 0:
        raise ValueError("reserved metadata field must be zero")
    return RTreeMetadata(order=order, root_page_id=root_page_id)


class RTreePageStore:
    """Vuelca/reconstruye un árbol en memoria, una página por nodo."""

    def __init__(self, disk_manager: DiskManager, buffer_manager: BufferManager) -> None:
        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager
        self._page_size = disk_manager.page_size
        # RTreeNode no es hashable (dataclass con eq), así que se indexa por id() y
        # se guarda el nodo para que su id no pueda reciclarse mientras siga mapeado.
        self._page_ids: dict[int, tuple[RTreeNode, int]] = {}

    def save_tree(self, tree: "RTree") -> None:
        self.save(tree.order, tree._root)

    def load_tree(self) -> "RTree":
        # Import local: rtree.py importará este módulo cuando use la persistencia paginada.
        from engine.indexes.rtree.rtree import RTree

        order, root = self.load()
        tree = RTree(order=order)
        tree._root = root
        return tree

    def save(self, order: int, root: RTreeNode | None) -> None:
        self._check_order(order)
        if self._disk_manager.page_count == 0:
            if self._disk_manager.allocate_page() != METADATA_PAGE_ID:
                raise ValueError("R-Tree metadata page must be page 0")

        page_ids: dict[int, tuple[RTreeNode, int]] = {}
        root_page_id = (
            EMPTY_ROOT_PAGE_ID if root is None else self._save_node(root, order, page_ids)
        )
        self._page_ids = page_ids
        self._write_page(
            METADATA_PAGE_ID,
            encode_metadata(self._page_size, RTreeMetadata(order, root_page_id)),
        )
        self._buffer_manager.flush_all()

    def load(self) -> tuple[int, RTreeNode | None]:
        if self._disk_manager.page_count == 0:
            raise ValueError("R-Tree file has no metadata page")
        metadata = decode_metadata(self._read_page(METADATA_PAGE_ID))
        self._check_order(metadata.order)

        page_ids: dict[int, tuple[RTreeNode, int]] = {}
        root = None
        if metadata.root_page_id != EMPTY_ROOT_PAGE_ID:
            if not self._is_node_page_id(metadata.root_page_id):
                raise ValueError(f"invalid root_page_id {metadata.root_page_id}")
            root = self._load_node(metadata.root_page_id, metadata.order, page_ids, set())
        self._page_ids = page_ids
        return metadata.order, root

    def _save_node(
        self,
        node: RTreeNode,
        order: int,
        page_ids: dict[int, tuple[RTreeNode, int]],
    ) -> int:
        if node.max_entries != order:
            raise ValueError(f"node max_entries {node.max_entries} does not match order {order}")
        page_id = self._page_id_for(node)
        page_ids[id(node)] = (node, page_id)

        if node.is_leaf:
            entries = tuple(node.entries)
        else:
            # Se persiste child.mbr y no el MBR de la tupla: el RTree solo mantiene
            # al día el MBR del hijo.
            entries = tuple(
                (child.mbr, self._save_node(child, order, page_ids)) for _mbr, child in node.entries
            )
        page = DecodedRTreePage(is_leaf=node.is_leaf, max_entries=order, entries=entries)
        self._write_page(page_id, _page_codec.encode_page(self._page_size, page))
        return page_id

    def _page_id_for(self, node: RTreeNode) -> int:
        known = self._page_ids.get(id(node))
        if known is not None and known[0] is node:
            return known[1]
        return self._disk_manager.allocate_page()

    def _load_node(
        self,
        page_id: int,
        order: int,
        page_ids: dict[int, tuple[RTreeNode, int]],
        visited: set[int],
    ) -> RTreeNode:
        visited.add(page_id)
        page = _page_codec.decode_page(self._read_page(page_id), self._page_size)
        if page.max_entries != order:
            raise ValueError(
                f"page {page_id} max_entries {page.max_entries} does not match order {order}"
            )

        if page.is_leaf:
            entries = list(page.entries)
        else:
            entries = []
            for stored_mbr, child_page_id in page.entries:
                if not self._is_node_page_id(child_page_id) or child_page_id in visited:
                    raise ValueError(f"invalid child_page_id {child_page_id} in page {page_id}")
                child = self._load_node(child_page_id, order, page_ids, visited)
                if not child.entries:
                    raise ValueError(f"child page {child_page_id} has no entries")
                if stored_mbr != child.mbr:
                    raise ValueError(f"stale MBR for child page {child_page_id}")
                entries.append((child.mbr, child))

        node = RTreeNode(is_leaf=page.is_leaf, entries=entries, max_entries=order)
        refresh_mbr(node)
        page_ids[id(node)] = (node, page_id)
        return node

    def _is_node_page_id(self, page_id: int) -> bool:
        return METADATA_PAGE_ID < page_id < self._disk_manager.page_count

    def _check_order(self, order: int) -> None:
        limit = min(
            _page_codec.leaf_capacity(self._page_size),
            _page_codec.internal_capacity(self._page_size),
        )
        if not _MIN_ORDER <= order <= limit:
            raise ValueError(
                f"order {order} incompatible with page_size {self._page_size} "
                f"(allowed [{_MIN_ORDER}, {limit}])"
            )

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
