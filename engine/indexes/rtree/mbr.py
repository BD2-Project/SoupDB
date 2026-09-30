"""Minimum Bounding Rectangle del R-Tree.

MBR es un rectángulo alineado a los ejes que agrupa puntos (hojas) o cajas de
hijos (nodos internos). Todas las operaciones son inmutables: devuelven un
nuevo MBR.
"""

from dataclasses import dataclass

from engine.indexes.rtree.point import Point


@dataclass(frozen=True)
class MBR:
    """Rectángulo mínimo con bordes inclusivos."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def from_point(cls, point: Point) -> "MBR":
        """MBR degenerado (área cero) de un punto."""
        return cls(point.x, point.y, point.x, point.y)

    def union(self, other: "MBR") -> "MBR":
        """MBR que cubre ambos rectángulos."""
        return MBR(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def area(self) -> float:
        """Área del rectángulo (cero para puntos)."""
        return max(0.0, self.max_x - self.min_x) * max(0.0, self.max_y - self.min_y)

    def enlargement(self, other: "MBR") -> float:
        """Incremento de área al cubrir ``other``."""
        return self.union(other).area() - self.area()

    def contains(self, point: Point) -> bool:
        """¿Contiene el punto (bordes inclusivos)?"""
        return self.min_x <= point.x <= self.max_x and self.min_y <= point.y <= self.max_y

    def intersects(self, other: "MBR") -> bool:
        """¿Se superpone con otro MBR (bordes inclusivos)?"""
        return not (
            other.max_x < self.min_x
            or other.min_x > self.max_x
            or other.max_y < self.min_y
            or other.min_y > self.max_y
        )
