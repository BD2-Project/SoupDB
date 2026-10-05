from __future__ import annotations

from collections.abc import Iterator

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import RTreeNode
from engine.indexes.rtree.point import Point


def iter_leaf_entries(
    root: RTreeNode | None,
) -> Iterator[tuple[Point, RID]]:
    if root is None:
        return

    stack = [root]

    while stack:
        node = stack.pop()

        if node.is_leaf:
            yield from node.entries
            continue

        children = [child for _stored_mbr, child in node.entries]
        stack.extend(reversed(children))


def iter_child_nodes(
    node: RTreeNode,
) -> Iterator[tuple[MBR, RTreeNode]]:
    if node.is_leaf:
        return

    for _stored_mbr, child in node.entries:
        if child.mbr is None:
            if child.entries:
                raise RuntimeError("non-empty R-Tree child without MBR")
            continue

        yield child.mbr, child
