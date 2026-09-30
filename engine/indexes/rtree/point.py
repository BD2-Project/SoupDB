"""Punto espacial: la clave del R-Tree.

Para poder pasar la conformance suite de ``Index`` (que usa claves escalares),
``from_key`` mapea claves 1D a puntos sobre el eje X; las claves string se
mapean a un punto estable por hash.
"""

import hashlib
from typing import NamedTuple


class Point(NamedTuple):
    """Coordenada (x, y) en el plano."""

    x: float
    y: float

    @classmethod
    def from_key(cls, key: object) -> "Point":
        """Normaliza una clave a un Point (soporta escalares 1D y tuplas 2D)."""
        if isinstance(key, Point):
            return key
        if isinstance(key, bool):
            raise ValueError("bool is not a valid R-Tree key")
        if isinstance(key, (int, float)):
            return cls(float(key), 0.0)
        if isinstance(key, str):
            digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            return cls(float(value & 0xFFFFFFFF), float((value >> 32) & 0xFFFFFFFF))
        if isinstance(key, tuple) and len(key) == 2:
            return cls(float(key[0]), float(key[1]))
        raise ValueError(f"unsupported R-Tree key {type(key).__name__}")
