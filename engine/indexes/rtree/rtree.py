"""Núcleo del R-Tree en memoria.

Estructura fija (capacidad por nodo = ``order``), inserción con elección de
subárbol por menor enlargement y **Quadratic Split**, actualización de MBRs,
búsqueda puntual y por rango, borrado simple (sin condense/reinsert) y una
interfaz de persistencia ``save``/``load``/``open`` (memoria <-> disco).

Es una implementación de :class:`engine.indexes.base.Index`; para cumplir la
conformance suite del gestor, las claves escalares se mapean a puntos sobre el
eje X (ver ``Point.from_key``).
"""

from pathlib import Path

from engine.common.rid import RID
from engine.indexes.base import Index, Key
from engine.indexes.rtree import _codec
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.node import (
    RTreeNode,
    entry_mbr,
    new_internal,
    new_leaf,
    refresh_mbr,
)
from engine.indexes.rtree.point import Point


class RTree(Index):
    """R-Tree en memoria con división cuadrática y persistencia por codec."""

    def __init__(self, order: int = 4) -> None:
        if order < 2:
            raise ValueError(f"order must be >= 2, got {order}")
        self._order = order
        self._closed = False
        self._root: RTreeNode | None = None

    @property
    def supports_range(self) -> bool:
        return True

    @property
    def order(self) -> int:
        return self._order

    # --- API del contrato Index ------------------------------------------

    def insert(self, key: Key, rid: RID) -> None:
        self._ensure_open()
        point = Point.from_key(key)
        if self._root is None:
            self._root = new_leaf(self._order, [(point, rid)])
            return
        promoted = self._insert_into(self._root, point, rid)
        if promoted is not None:
            self._root = new_internal(
                self._order,
                [(self._root.mbr, self._root), (promoted.mbr, promoted)],
            )

    def search(self, key: Key) -> list[RID]:
        self._ensure_open()
        point = Point.from_key(key)
        return self._search(self._root, point)

    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        self._ensure_open()
        lo_point = Point.from_key(lo)
        hi_point = Point.from_key(hi)
        box = MBR(
            min(lo_point.x, hi_point.x),
            min(lo_point.y, hi_point.y),
            max(lo_point.x, hi_point.x),
            max(lo_point.y, hi_point.y),
        )
        return self._range(self._root, box)

    def remove(self, key: Key, rid: RID | None = None) -> int:
        self._ensure_open()
        point = Point.from_key(key)
        removed = self._remove(self._root, point, rid)
        if self._root is not None and not self._root.entries:
            self._root = None
        return removed

    def close(self) -> None:
        self._closed = True

    # --- Persistencia memoria <-> disco ----------------------------------

    def save(self, path: str | Path) -> None:
        """Serializa el árbol completo a un archivo binario."""
        Path(path).write_bytes(_codec.encode_tree(self._order, self._root))

    @classmethod
    def load(cls, path: str | Path) -> "RTree":
        """Carga un árbol persistido."""
        order, root = _codec.decode_tree(Path(path).read_bytes())
        tree = cls(order=order)
        tree._root = root
        return tree

    @classmethod
    def open(cls, path: str | Path, *, order: int = 4) -> "RTree":
        """Abre (o crea) un R-Tree persistido en ``path``."""
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls(order=order)

    # --- internals --------------------------------------------------------

    def _insert_into(self, node: RTreeNode, point: Point, rid: RID) -> RTreeNode | None:
        if node.is_leaf:
            node.entries.append((point, rid))
            refresh_mbr(node)
            if len(node.entries) > node.max_entries:
                return self._split(node)
            return None

        child = self._choose_child(node, point)
        promoted = self._insert_into(child, point, rid)
        # El MBR del hijo pudo crecer aunque no haya habido split.
        refresh_mbr(node)
        if promoted is not None:
            node.entries.append((promoted.mbr, promoted))
            refresh_mbr(node)
            if len(node.entries) > node.max_entries:
                return self._split(node)
        return None

    def _choose_child(self, internal: RTreeNode, point: Point) -> RTreeNode:
        target = MBR.from_point(point)
        best_entry = None
        best_key: tuple[float, float] | None = None
        for _child_mbr, child in internal.entries:
            key = (child.mbr.enlargement(target), child.mbr.area())
            if best_key is None or key < best_key:
                best_key = key
                best_entry = child
        if best_entry is None:
            raise RuntimeError("internal node without children")
        return best_entry

    def _split(self, node: RTreeNode) -> RTreeNode:
        """Quadratic Split: muta ``node`` a la mitad izquierda y devuelve la derecha."""
        entries = list(node.entries)
        min_entries = max(1, node.max_entries // 2)

        first, second = self._pick_seeds(node, entries)
        left_entries = [first]
        right_entries = [second]
        remaining = [entry for entry in entries if entry is not first and entry is not second]

        left_box = entry_mbr(node, first)
        right_box = entry_mbr(node, second)

        while remaining:
            if len(left_entries) + len(remaining) == min_entries:
                for entry in remaining:
                    left_box = left_box.union(entry_mbr(node, entry))
                    left_entries.append(entry)
                break
            if len(right_entries) + len(remaining) == min_entries:
                for entry in remaining:
                    right_box = right_box.union(entry_mbr(node, entry))
                    right_entries.append(entry)
                break

            best = None
            for entry in remaining:
                box = entry_mbr(node, entry)
                delta_left = left_box.enlargement(box)
                delta_right = right_box.enlargement(box)
                difference = abs(delta_left - delta_right)
                if best is None or difference > best[0]:
                    best = (difference, entry, box, delta_left, delta_right)

            _, entry, box, delta_left, delta_right = best
            if delta_left < delta_right or (
                delta_left == delta_right and left_box.area() <= right_box.area()
            ):
                left_box = left_box.union(box)
                left_entries.append(entry)
            else:
                right_box = right_box.union(box)
                right_entries.append(entry)
            remaining.remove(entry)

        node.entries = left_entries
        node.mbr = left_box
        right = RTreeNode(
            is_leaf=node.is_leaf,
            entries=right_entries,
            max_entries=node.max_entries,
            mbr=right_box,
        )
        return right

    def _pick_seeds(self, node: RTreeNode, entries: list) -> tuple:
        best = None
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                box_i = entry_mbr(node, entries[i])
                box_j = entry_mbr(node, entries[j])
                wasted = box_i.union(box_j).area() - box_i.area() - box_j.area()
                if best is None or wasted > best[0]:
                    best = (wasted, entries[i], entries[j])
        if best is None:
            raise RuntimeError("cannot pick seeds from empty node")
        return best[1], best[2]

    def _search(self, node: RTreeNode | None, point: Point) -> list[RID]:
        if node is None:
            return []
        if node.is_leaf:
            return [rid for stored, rid in node.entries if stored == point]
        result: list[RID] = []
        for _child_mbr, child in node.entries:
            if child.mbr.contains(point):
                result.extend(self._search(child, point))
        return result

    def _range(self, node: RTreeNode | None, box: MBR) -> list[RID]:
        if node is None:
            return []
        if node.is_leaf:
            return [rid for stored, rid in node.entries if box.contains(stored)]
        result: list[RID] = []
        for _child_mbr, child in node.entries:
            if child.mbr.intersects(box):
                result.extend(self._range(child, box))
        return result

    def _remove(self, node: RTreeNode | None, point: Point, rid: RID | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            removed = 0
            removed_one = False
            keep: list = []
            for stored, stored_rid in node.entries:
                if stored == point and (rid is None or (not removed_one and stored_rid == rid)):
                    removed += 1
                    removed_one = True
                else:
                    keep.append((stored, stored_rid))
            node.entries = keep
            refresh_mbr(node)
            return removed
        removed = 0
        keep: list = []
        for entry in node.entries:
            _child_mbr, child = entry
            if child.mbr is not None and child.mbr.contains(point):
                removed += self._remove(child, point, rid)
            if not child.entries:
                # Poda de nodos vacíos (hojas vaciadas o subárboles sin entradas).
                continue
            keep.append(entry)
        node.entries = keep
        refresh_mbr(node)
        return removed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("R-Tree is closed")
