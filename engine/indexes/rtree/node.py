"""Nodos del R-Tree: estructura fija con capacidad acotada.

Un nodo hoja guarda ``(Point, RID)``; un nodo interno guarda ``(MBR, RTreeNode)``.
Cada nodo mantiene su propio MBR (actualizado tras mutaciones) para poder podar
en búsquedas y elegir subárbol en inserciones.
"""

from dataclasses import dataclass
from typing import Any

from engine.common.rid import RID
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point


@dataclass
class RTreeNode:
    """Nodo del R-Tree (hoja o interno)."""

    is_leaf: bool
    entries: list[tuple[Any, Any]]
    max_entries: int
    mbr: MBR | None = None


def entry_mbr(node: RTreeNode, entry: tuple[Any, Any]) -> MBR:
    """MBR de una entrada: punto (hoja) o el MBR actual del hijo (interno).

    En nodos internos se usa ``child.mbr`` (no el valor almacenado en la
    entrada) para que el MBR esté siempre al día tras las mutaciones.
    """
    if node.is_leaf:
        point, _rid = entry
        return MBR.from_point(point)
    _child_mbr, child = entry
    return child.mbr


def compute_mbr(node: RTreeNode) -> MBR | None:
    """Recalcula el MBR de un nodo desde sus entradas."""
    if not node.entries:
        return None
    mbr = entry_mbr(node, node.entries[0])
    for entry in node.entries[1:]:
        mbr = mbr.union(entry_mbr(node, entry))
    return mbr


def refresh_mbr(node: RTreeNode) -> None:
    """Actualiza el MBR del nodo en el lugar."""
    node.mbr = compute_mbr(node)


def new_leaf(max_entries: int, entries: list[tuple[Point, RID]] | None = None) -> RTreeNode:
    node = RTreeNode(is_leaf=True, entries=list(entries or []), max_entries=max_entries)
    refresh_mbr(node)
    return node


def new_internal(
    max_entries: int,
    entries: list[tuple[MBR, RTreeNode]] | None = None,
) -> RTreeNode:
    node = RTreeNode(is_leaf=False, entries=list(entries or []), max_entries=max_entries)
    refresh_mbr(node)
    return node
