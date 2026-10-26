"""Bag of words sobre descriptores locales: codebook K-Means, histograma y tf-idf.

No depende del tipo de medio: recibe descriptores (una fila por descriptor), así
que imagen (SIFT) y audio (MFCC) comparten estas mismas funciones.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def sample_descriptors(
    descriptor_sets: Iterable[np.ndarray], max_descriptors: int | None, seed: int = 0
) -> np.ndarray:
    """Muestra uniforme de como máximo ``max_descriptors`` filas, leyendo un bloque cada vez.

    Reservoir sampling (algoritmo R): no necesita saber cuántos descriptores vendrán
    y nunca guarda más de ``max_descriptors`` filas. Con ``None`` devuelve todos.
    """
    if max_descriptors is None:
        return np.vstack(list(descriptor_sets))
    if max_descriptors <= 0:
        raise ValueError(f"max_descriptors must be positive, got {max_descriptors}")

    rng = np.random.default_rng(seed)
    sample = None
    seen = 0
    for block in descriptor_sets:
        if sample is None:
            sample = np.empty((max_descriptors, block.shape[1]), dtype=block.dtype)
        # Mientras la muestra no esté llena, entra todo.
        fill = min(len(block), max(0, max_descriptors - seen))
        sample[seen : seen + fill] = block[:fill]
        # El descriptor número t (contando desde 1) se queda con probabilidad M/t:
        # se sortea un hueco en [0, t) y, si cae dentro de la muestra, reemplaza a ese.
        positions = np.arange(seen + fill + 1, seen + len(block) + 1)
        slots = rng.integers(0, positions)
        for i in np.flatnonzero(slots < max_descriptors):
            sample[slots[i]] = block[fill + i]
        seen += len(block)

    if sample is None:
        return np.empty((0, 0), dtype=np.float32)
    return sample[: min(seen, max_descriptors)]


def train_codebook(descriptors: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    if len(descriptors) < k:
        raise ValueError(f"need at least {k} descriptors for {k} words, got {len(descriptors)}")
    kmeans = KMeans(n_clusters=k, random_state=seed)
    # Con varios hilos OpenMP suma en distinto orden y los centroides cambian en
    # los últimos decimales; un hilo hace que la misma semilla dé el mismo codebook.
    with threadpool_limits(limits=1):
        kmeans.fit(descriptors)
    return kmeans.cluster_centers_.astype(np.float32)


def assign_words(descriptors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Palabra visual de cada descriptor = índice del centroide más cercano (euclidiana).
    words = [np.argmin(np.linalg.norm(centroids - d, axis=1)) for d in descriptors]
    return np.array(words, dtype=np.intp)


def count_words(words: np.ndarray, k: int) -> np.ndarray:
    # Con k=4: palabras [0, 2, 2, 1, 2] -> histograma [1, 1, 3, 0].
    return np.bincount(words, minlength=k)


def compute_idf(histograms: np.ndarray) -> np.ndarray:
    # histograms: una fila por imagen de la colección, una columna por palabra.
    n = len(histograms)
    df = (histograms > 0).sum(axis=0)
    return np.log((n + 1) / (df + 1)) + 1


def tfidf_vector(histogram: np.ndarray, idf: np.ndarray) -> list[float]:
    weighted = histogram * idf
    length = np.linalg.norm(weighted)
    if length == 0:
        # Sin descriptores no hay nada que normalizar: queda el vector cero.
        return weighted.tolist()
    return (weighted / length).tolist()


def save_codebook(path: str | Path, centroids: np.ndarray, idf: np.ndarray) -> None:
    np.savez(path, centroids=centroids, idf=idf)


def load_codebook(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["centroids"], data["idf"]
