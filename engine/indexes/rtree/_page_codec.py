"""Codec binario de página para nodos R-Tree (un nodo por página).

Formato (big-endian), la página ocupa exactamente ``page_size`` bytes:

.. code-block::

    header  : magic "RN" (2B) | version u8 | node_type u8 (0 interno, 1 hoja)
              | entry_count u16 | max_entries u16 | reserved u64 = 0
    hoja    : por entrada point x,y (f64 f64) + rid page_id,slot (q q)
    interna : por entrada mbr min_x,min_y,max_x,max_y (f64 x4) + child_page_id (q)
    resto   : ceros
"""

import struct
from dataclasses import dataclass
from typing import Any

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point

MAGIC = b"RN"
VERSION = 1

NODE_TYPE_INTERNAL = 0
NODE_TYPE_LEAF = 1

_HEADER = struct.Struct(">2sBBHHQ")
_LEAF_ENTRY = struct.Struct(">ddqq")
_INTERNAL_ENTRY = struct.Struct(">ddddq")

HEADER_SIZE = _HEADER.size
LEAF_ENTRY_SIZE = _LEAF_ENTRY.size
INTERNAL_ENTRY_SIZE = _INTERNAL_ENTRY.size

# Igual que RTree(order >= 2): un nodo con menos de 2 entradas no puede dividirse.
_MIN_MAX_ENTRIES = 2


@dataclass(frozen=True)
class DecodedRTreePage:
    """Nodo tal como vive en una página.

    Hoja: ``entries`` son ``(Point, RID)``. Interno: ``(MBR, child_page_id)``.
    """

    is_leaf: bool
    max_entries: int
    entries: tuple[tuple[Any, Any], ...]


def leaf_capacity(page_size: int) -> int:
    return max(0, (page_size - HEADER_SIZE) // LEAF_ENTRY_SIZE)


def internal_capacity(page_size: int) -> int:
    return max(0, (page_size - HEADER_SIZE) // INTERNAL_ENTRY_SIZE)


def encode_page(page_size: int, page: DecodedRTreePage) -> bytes:
    _check_page_size(page_size)
    _check_counts(page_size, len(page.entries), page.max_entries)

    out = bytearray(page_size)
    _HEADER.pack_into(
        out,
        0,
        MAGIC,
        VERSION,
        NODE_TYPE_LEAF if page.is_leaf else NODE_TYPE_INTERNAL,
        len(page.entries),
        page.max_entries,
        0,
    )
    offset = HEADER_SIZE
    if page.is_leaf:
        for point, rid in page.entries:
            _LEAF_ENTRY.pack_into(out, offset, point.x, point.y, rid.page_id, rid.slot)
            offset += LEAF_ENTRY_SIZE
    else:
        for mbr, child_page_id in page.entries:
            _check_child_page_id(child_page_id)
            _INTERNAL_ENTRY.pack_into(
                out, offset, mbr.min_x, mbr.min_y, mbr.max_x, mbr.max_y, child_page_id
            )
            offset += INTERNAL_ENTRY_SIZE
    return bytes(out)


def decode_page(data: bytes, page_size: int) -> DecodedRTreePage:
    _check_page_size(page_size)
    if len(data) != page_size:
        raise ValueError(f"invalid page size: expected {page_size} bytes, got {len(data)}")

    magic, version, node_type, count, max_entries, reserved = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"invalid R-Tree page magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported R-Tree page version {version}")
    if node_type not in (NODE_TYPE_INTERNAL, NODE_TYPE_LEAF):
        raise ValueError(f"invalid R-Tree node type {node_type}")
    if reserved != 0:
        raise ValueError("reserved header field must be zero")
    is_leaf = node_type == NODE_TYPE_LEAF
    _check_counts(page_size, count, max_entries)

    entries: list[tuple[Any, Any]] = []
    offset = HEADER_SIZE
    if is_leaf:
        for _ in range(count):
            x, y, page_id, slot = _LEAF_ENTRY.unpack_from(data, offset)
            entries.append((Point(x, y), RID(page_id, slot)))
            offset += LEAF_ENTRY_SIZE
    else:
        for _ in range(count):
            min_x, min_y, max_x, max_y, child_page_id = _INTERNAL_ENTRY.unpack_from(data, offset)
            _check_child_page_id(child_page_id)
            entries.append((MBR(min_x, min_y, max_x, max_y), child_page_id))
            offset += INTERNAL_ENTRY_SIZE
    return DecodedRTreePage(is_leaf=is_leaf, max_entries=max_entries, entries=tuple(entries))


def _check_page_size(page_size: int) -> None:
    if internal_capacity(page_size) < _MIN_MAX_ENTRIES:
        raise ValueError(f"page_size {page_size} cannot hold an R-Tree node")


def _check_counts(page_size: int, count: int, max_entries: int) -> None:
    # max_entries es el order global del RTree (hojas e internos comparten order),
    # así que se acota por la capacidad del tipo de nodo más restrictivo.
    capacity = min(leaf_capacity(page_size), internal_capacity(page_size))
    if not _MIN_MAX_ENTRIES <= max_entries <= capacity:
        raise ValueError(f"max_entries {max_entries} out of range [{_MIN_MAX_ENTRIES}, {capacity}]")
    if count > max_entries:
        raise ValueError(f"entry count {count} exceeds max_entries {max_entries}")


def _check_child_page_id(child_page_id: int) -> None:
    if child_page_id < 0:
        raise ValueError(f"child_page_id must be non-negative, got {child_page_id}")
