"""Vector de un audio: MFCC -> palabras acústicas -> histograma -> tf-idf -> L2.

Para el dataset, cada pista se parte en segmentos consecutivos de duración fija
(5 s por defecto, sin solapamiento) y cada segmento es un objeto independiente,
identificado como ``ruta#segment=000000``.
"""

from collections.abc import Iterator, Sequence
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
from engine.multimedia.descriptor_cache import cache_path, load_array, save_array

SAMPLE_RATE = 22050
N_MFCC = 20
SEGMENT_SECONDS = 5.0
# El último segmento se conserva si dura al menos esta fracción de SEGMENT_SECONDS.
MIN_SEGMENT_RATIO = 0.5


def load_audio(path: str | Path) -> np.ndarray:
    # load mezcla los canales a mono y remuestrea a SAMPLE_RATE en un solo paso.
    signal, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return signal


def mfcc_from_signal(signal: np.ndarray, n_mfcc: int = N_MFCC) -> np.ndarray:
    # Única fórmula MFCC: la usan el archivo completo y cada segmento.
    if len(signal) == 0:
        # Audio vacío: cero descriptores (librosa inventaría un frame de relleno).
        return np.empty((0, n_mfcc), dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
    # librosa devuelve (n_mfcc, T); transponemos a una fila por frame, como SIFT.
    return mfcc.T


def extract_mfcc(path: str | Path, n_mfcc: int = N_MFCC) -> np.ndarray:
    return mfcc_from_signal(load_audio(path), n_mfcc)


def segment_id(path: str | Path, segment: int) -> str:
    return f"{Path(path).as_posix()}#segment={segment:06d}"


def _split(signal: np.ndarray, duration: float) -> list[np.ndarray]:
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    size = round(duration * SAMPLE_RATE)
    # Pista de 18 s con duration=5: [0-5), [5-10), [10-15), [15-18]. Son vistas sobre la
    # señal: no se copian muestras ni se rellena nada.
    segments = [signal[start : start + size] for start in range(0, len(signal), size)]
    # Solo el último puede ser más corto. Se queda si dura al menos la mitad (3 s sí);
    # una cola de decodificación (0,1 s, 60 muestras) no es un objeto del dataset.
    if segments and len(segments[-1]) < MIN_SEGMENT_RATIO * size:
        segments.pop()
    return segments


def iter_audio_segments(
    path: str | Path, duration: float = SEGMENT_SECONDS
) -> Iterator[tuple[int, np.ndarray]]:
    yield from enumerate(_split(load_audio(path), duration))


def count_audio_segments(path: str | Path, duration: float = SEGMENT_SECONDS) -> int:
    # Mismo corte que iter_audio_segments, así el manifest nunca discrepa de la extracción.
    return len(_split(load_audio(path), duration))


def _segment_cache_path(
    cache_dir: str | Path, path: str | Path, segment: int, duration: float, n_mfcc: int
) -> Path:
    params = {
        "sample_rate": SAMPLE_RATE,
        "n_mfcc": n_mfcc,
        "duration": duration,
        "segment": segment,
    }
    return cache_path(cache_dir, "audio_segment", path, params)


def extract_mfcc_cached(
    path: str | Path,
    segment: int,
    cache_dir: str | Path,
    duration: float = SEGMENT_SECONDS,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    entry = _segment_cache_path(cache_dir, path, segment, duration, n_mfcc)
    mfcc = load_array(entry)
    if mfcc is not None:
        return mfcc
    # Fallo de caché: se decodifica la pista una vez y se guardan todos sus segmentos,
    # así los siguientes segmentos de la misma pista ya salen de la caché.
    segments = _split(load_audio(path), duration)
    if not 0 <= segment < len(segments):
        raise IndexError(f"{path} has {len(segments)} segments, asked for {segment}")
    for index, samples in enumerate(segments):
        computed = mfcc_from_signal(samples, n_mfcc)
        save_array(_segment_cache_path(cache_dir, path, index, duration, n_mfcc), computed)
        if index == segment:
            mfcc = computed
    return mfcc


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
