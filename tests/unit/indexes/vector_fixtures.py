"""Synthetic vectors shared by the vector index tests."""

import random

from engine.common.rid import RID


def clustered_vectors(
    n: int, dim: int, *, clusters: int = 10, spread: float = 0.15, seed: int = 0
) -> list[tuple[list[float], RID]]:
    """Gaussian blobs around random centers: closer to real descriptors than uniform noise."""
    rng = random.Random(seed)
    centers = [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(clusters)]
    items = []
    for i in range(n):
        center = centers[i % clusters]
        items.append(([c + rng.gauss(0, spread) for c in center], RID(i // 50, i % 50)))
    return items


def recall_at_k(found: list[RID], expected: list[RID]) -> float:
    return len(set(found) & set(expected)) / len(expected) if expected else 1.0
