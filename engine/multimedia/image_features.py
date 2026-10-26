"""Vector de una imagen: SIFT -> palabras visuales -> histograma -> tf-idf -> L2."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from engine.multimedia.codebook import (
    assign_words,
    compute_idf,
    count_words,
    tfidf_vector,
    train_codebook,
)

SIFT_DIMENSION = 128


def extract_sift(path: str | Path) -> np.ndarray:
    # imdecode en vez de imread: imread no abre rutas con tildes en Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    if descriptors is None:
        # Imagen sin keypoints (p. ej. un color plano): cero descriptores.
        return np.empty((0, SIFT_DIMENSION), dtype=np.float32)
    return descriptors


def train_image_codebook(
    paths: Sequence[str | Path], k: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    descriptors = [extract_sift(path) for path in paths]
    centroids = train_codebook(np.vstack(descriptors), k, seed)
    histograms = [count_words(assign_words(d, centroids), k) for d in descriptors]
    idf = compute_idf(np.array(histograms))
    return centroids, idf


def extract_image_vector(path: str | Path, centroids: np.ndarray, idf: np.ndarray) -> list[float]:
    descriptors = extract_sift(path)
    words = assign_words(descriptors, centroids)
    histogram = count_words(words, len(centroids))
    return tfidf_vector(histogram, idf)
