"""K-Means (Lloyd + k-means++) and reservoir sampling, in pure Python.

numpy is only allowed for feature extraction, never inside a data structure,
so the IVF trains its own centroids. Training runs over a bounded sample, so
its memory does not grow with the collection.
"""

import math
import random
from collections.abc import Iterable, Sequence

Vector = tuple[float, ...]


def reservoir_sample(
    vectors: Iterable[Sequence[float]], size: int, rng: random.Random
) -> list[Vector]:
    """Uniform sample of at most ``size`` vectors in one streaming pass (algorithm R)."""
    sample: list[Vector] = []
    for seen, vector in enumerate(vectors):
        if seen < size:
            sample.append(tuple(vector))
            continue
        slot = rng.randrange(seen + 1)
        if slot < size:
            sample[slot] = tuple(vector)
    return sample


def nearest(vector: Sequence[float], centroids: Sequence[Sequence[float]]) -> int:
    best = 0
    best_distance = math.inf
    for position, centroid in enumerate(centroids):
        distance = math.dist(vector, centroid)
        if distance < best_distance:
            best, best_distance = position, distance
    return best


def kmeans(
    sample: Sequence[Vector],
    k: int,
    *,
    iterations: int,
    rng: random.Random,
) -> list[Vector]:
    """Return ``min(k, len(sample))`` centroids trained with Lloyd's algorithm."""
    if not sample:
        raise ValueError("cannot train k-means on an empty sample")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(sample))
    dim = len(sample[0])

    centroids = _kmeans_plus_plus(sample, k, rng)
    assignment = [-1] * len(sample)
    for _ in range(iterations):
        changed = False
        sums = [[0.0] * dim for _ in range(k)]
        sizes = [0] * k
        for position, vector in enumerate(sample):
            cluster = nearest(vector, centroids)
            if cluster != assignment[position]:
                assignment[position] = cluster
                changed = True
            sizes[cluster] += 1
            accumulator = sums[cluster]
            for axis, value in enumerate(vector):
                accumulator[axis] += value

        for cluster in range(k):
            if sizes[cluster]:
                centroids[cluster] = tuple(total / sizes[cluster] for total in sums[cluster])
            else:
                # Un cluster vacío se resiembra en un punto al azar para no perder una lista.
                centroids[cluster] = sample[rng.randrange(len(sample))]
        if not changed:
            break
    return centroids


def _kmeans_plus_plus(sample: Sequence[Vector], k: int, rng: random.Random) -> list[Vector]:
    """Seed centroids far apart, so Lloyd converges in fewer iterations."""
    centroids = [sample[rng.randrange(len(sample))]]
    closest = [math.dist(vector, centroids[0]) ** 2 for vector in sample]
    while len(centroids) < k:
        total = sum(closest)
        if total == 0:
            centroids.append(sample[rng.randrange(len(sample))])
            continue
        target = rng.uniform(0, total)
        cumulative = 0.0
        chosen = len(sample) - 1
        for position, weight in enumerate(closest):
            cumulative += weight
            if cumulative >= target:
                chosen = position
                break
        centroid = sample[chosen]
        centroids.append(centroid)
        for position, vector in enumerate(sample):
            distance = math.dist(vector, centroid) ** 2
            if distance < closest[position]:
                closest[position] = distance
    return centroids
