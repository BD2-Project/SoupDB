"""Codec binario del R-Tree: serialización memoria <-> disco.

Formato (big-endian):

.. code-block::

    magic "RT" (2B) | version 0x01 (1B) | order u16 BE (2B) | has_root (1B)
    [nodo]  is_leaf (1B) | count u32 BE (4B)
            hoja   : por entrada point x,y (f64 f64) + rid page,slot (q q)
            interna: por entrada mbr (f64 f64 f64 f64) + hijo (recursivo)
"""

import struct

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import RTreeNode, refresh_mbr
from engine.indexes.rtree.point import Point

MAGIC = b"RT"
VERSION = 1

_HEADER_SIZE = 6  # magic(2) + version(1) + order(2) + has_root(1)


def encode_tree(order: int, root: RTreeNode | None) -> bytes:
    """Serializa el árbol completo."""
    out = bytearray()
    out += MAGIC
    out += bytes((VERSION,))
    out += struct.pack(">H", order)
    if root is None:
        out += bytes((0,))
    else:
        out += bytes((1,))
        _encode_node(out, root)
    return bytes(out)


def _encode_node(out: bytearray, node: RTreeNode) -> None:
    out += bytes((1 if node.is_leaf else 0,))
    out += struct.pack(">I", len(node.entries))
    if node.is_leaf:
        for point, rid in node.entries:
            out += struct.pack(">dd", point.x, point.y)
            out += struct.pack(">qq", rid.page_id, rid.slot)
    else:
        for child_mbr, child in node.entries:
            out += struct.pack(
                ">dddd",
                child_mbr.min_x,
                child_mbr.min_y,
                child_mbr.max_x,
                child_mbr.max_y,
            )
            _encode_node(out, child)


def decode_tree(data: bytes) -> tuple[int, RTreeNode | None]:
    """Decodifica un árbol persistido."""
    if len(data) < _HEADER_SIZE or data[:2] != MAGIC:
        raise ValueError("invalid R-Tree file")
    if data[2] != VERSION:
        raise ValueError(f"unsupported R-Tree version {data[2]}")
    (order,) = struct.unpack_from(">H", data, 3)
    offset = 5
    has_root = data[offset]
    offset += 1
    if not has_root:
        return order, None
    root, _offset = _decode_node(data, offset, order)
    return order, root


def _decode_node(data: bytes, offset: int, order: int) -> tuple[RTreeNode, int]:
    is_leaf = data[offset] == 1
    offset += 1
    (count,) = struct.unpack_from(">I", data, offset)
    offset += 4
    if is_leaf:
        entries: list = []
        for _ in range(count):
            x, y = struct.unpack_from(">dd", data, offset)
            offset += 16
            page_id, slot = struct.unpack_from(">qq", data, offset)
            offset += 16
            entries.append((Point(x, y), RID(page_id, slot)))
    else:
        entries = []
        for _ in range(count):
            min_x, min_y, max_x, max_y = struct.unpack_from(">dddd", data, offset)
            offset += 32
            child, offset = _decode_node(data, offset, order)
            entries.append((MBR(min_x, min_y, max_x, max_y), child))
    node = RTreeNode(is_leaf=is_leaf, entries=entries, max_entries=order)
    refresh_mbr(node)
    return node, offset
