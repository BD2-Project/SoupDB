"""Bag of words sobre descriptores locales: codebook K-Means, histograma y tf-idf.

No depende del tipo de medio: recibe descriptores (una fila por descriptor), así
que imagen (SIFT) y audio (MFCC) comparten estas mismas funciones.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def sample_descriptors(
    descriptor_sets: Sequence[np.ndarray], max_descriptors: int, seed: int = 0
) -> np.ndarray:
    # Solo acota lo que recibe K-Means; histogramas e IDF siguen usando todos los descriptores.
    if max_descriptors <= 0:
        raise ValueError(f"max_descriptors must be positive, got {max_descriptors}")
    descriptors = np.vstack(descriptor_sets)
    if len(descriptors) <= max_descriptors:
        return descriptors
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(descriptors), size=max_descriptors, replace=False)
    return descriptors[chosen]


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
