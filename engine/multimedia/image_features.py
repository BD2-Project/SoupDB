"""Vector de una imagen: SIFT -> palabras visuales -> histograma -> tf-idf -> L2.

Antes de SIFT: imagen grande -> resize proporcional -> grayscale -> SIFT.
"""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from engine.multimedia.codebook import (
    assign_words,
    compute_idf,
    count_words,
    sample_descriptors,
    tfidf_vector,
    train_codebook,
)
from engine.multimedia.descriptor_cache import cache_path, cached

SIFT_DIMENSION = 128
# Lado mayor máximo antes de SIFT: acota el coste por imagen (más píxeles, más keypoints).
MAX_IMAGE_SIDE = 1024


def resize_to_max_side(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        # Ya cabe: nunca se agranda.
        return image
    # Un único factor para ancho y alto: se conserva el aspect ratio.
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # INTER_AREA promedia los píxeles que se juntan: es el filtro indicado para reducir.
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def extract_sift(path: str | Path, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    # imdecode en vez de imread: imread no abre rutas con tildes en Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image {path}")
    image = resize_to_max_side(image, max_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    if descriptors is None:
        # Imagen sin keypoints (p. ej. un color plano): cero descriptores.
        return np.empty((0, SIFT_DIMENSION), dtype=np.float32)
    return descriptors


def extract_sift_cached(
    path: str | Path, cache_dir: str | Path, max_side: int = MAX_IMAGE_SIDE
) -> np.ndarray:
    # max_side entra en la clave: otro tamaño da otros descriptores y otra entrada.
    entry = cache_path(cache_dir, "image", path, {"extractor": "sift", "max_side": max_side})
    return cached(entry, lambda: extract_sift(path, max_side))


def train_image_codebook(
    paths: Sequence[str | Path], k: int, seed: int = 0, max_descriptors: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # 1ª pasada: cada archivo se extrae, alimenta la muestra y se libera.
    training = sample_descriptors((extract_sift(path) for path in paths), max_descriptors, seed)
    centroids = train_codebook(training, k, seed)
    # 2ª pasada: se vuelve a extraer cada archivo para su histograma completo.
    histograms = [count_words(assign_words(extract_sift(path), centroids), k) for path in paths]
    idf = compute_idf(np.array(histograms))
    return centroids, idf


def extract_image_vector(path: str | Path, centroids: np.ndarray, idf: np.ndarray) -> list[float]:
    descriptors = extract_sift(path)
    words = assign_words(descriptors, centroids)
    histogram = count_words(words, len(centroids))
    return tfidf_vector(histogram, idf)
