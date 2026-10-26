"""Vector de un audio: MFCC -> palabras acústicas -> histograma -> tf-idf -> L2."""

from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np

from engine.multimedia.codebook import (
    assign_words,
    compute_idf,
    count_words,
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
    paths: Sequence[str | Path], k: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    descriptors = [extract_mfcc(path) for path in paths]
    centroids = train_codebook(np.vstack(descriptors), k, seed)
    histograms = [count_words(assign_words(d, centroids), k) for d in descriptors]
    idf = compute_idf(np.array(histograms))
    return centroids, idf


def extract_audio_vector(path: str | Path, centroids: np.ndarray, idf: np.ndarray) -> list[float]:
    descriptors = extract_mfcc(path)
    words = assign_words(descriptors, centroids)
    histogram = count_words(words, len(centroids))
    return tfidf_vector(histogram, idf)
