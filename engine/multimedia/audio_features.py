"""Vector de un audio: MFCC -> palabras acústicas -> histograma -> tf-idf -> L2."""

from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np

from engine.multimedia.codebook import (
    assign_words,
    compute_idf,
    count_words,
    sample_descriptors,
    tfidf_vector,
    train_codebook,
)

SAMPLE_RATE = 22050
N_MFCC = 20


def extract_mfcc(path: str | Path, n_mfcc: int = N_MFCC) -> np.ndarray:
    # load mezcla los canales a mono y remuestrea a SAMPLE_RATE en un solo paso.
    signal, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(signal) == 0:
        # Audio vacío: cero descriptores (librosa inventaría un frame de relleno).
        return np.empty((0, n_mfcc), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    # librosa devuelve (n_mfcc, T); transponemos a una fila por frame, como SIFT.
    return mfcc.T


def train_audio_codebook(
    paths: Sequence[str | Path], k: int, seed: int = 0, max_descriptors: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # 1ª pasada: cada archivo se extrae, alimenta la muestra y se libera.
    training = sample_descriptors((extract_mfcc(path) for path in paths), max_descriptors, seed)
    centroids = train_codebook(training, k, seed)
    # 2ª pasada: se vuelve a extraer cada archivo para su histograma completo.
    histograms = [count_words(assign_words(extract_mfcc(path), centroids), k) for path in paths]
    idf = compute_idf(np.array(histograms))
    return centroids, idf


def extract_audio_vector(path: str | Path, centroids: np.ndarray, idf: np.ndarray) -> list[float]:
    descriptors = extract_mfcc(path)
    words = assign_words(descriptors, centroids)
    histogram = count_words(words, len(centroids))
    return tfidf_vector(histogram, idf)
