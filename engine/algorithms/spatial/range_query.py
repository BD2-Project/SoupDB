from __future__ import annotations

from collections.abc import Iterator

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point

from .traversal import iter_child_nodes
from .validation import validate_mbr


def iter_range_entries(
    root: RTreeNode | None,
    box: MBR,
) -> Iterator[tuple[Point, RID]]:
    box = validate_mbr(box)

    if root is None or root.mbr is None:
        return

    if not root.mbr.intersects(box):
        return

    stack = [root]

    while stack:
        node = stack.pop()

        if node.is_leaf:
            for point, rid in node.entries:
                if box.contains(point):
                    yield point, rid
            continue

        children = [
            child for child_mbr, child in iter_child_nodes(node) if child_mbr.intersects(box)
        ]
        stack.extend(reversed(children))
