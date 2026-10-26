import gc
import math
import weakref

import numpy as np
import pytest
import soundfile as sf

from engine.indexes.vector.metrics import euclidean_distance
from engine.multimedia import audio_features
from engine.multimedia.audio_features import (
    N_MFCC,
    SAMPLE_RATE,
    extract_audio_vector,
    extract_mfcc,
    train_audio_codebook,
)
from engine.multimedia.codebook import assign_words, count_words
from engine.multimedia.vector_file import VectorRecord, iter_vectors, write_vectors

K = 8


def _write_tone(path, frequency, seconds=1.0, sample_rate=SAMPLE_RATE, channels=1):
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, np.stack([tone] * channels, axis=1), sample_rate)
    return path


@pytest.fixture(scope="module")
def audios(tmp_path_factory):
    folder = tmp_path_factory.mktemp("audios")
    # Un acorde para que haya frames variados, además de los dos tonos puros.
    chord = folder / "chord.wav"
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    sf.write(chord, 0.3 * (np.sin(2 * np.pi * 330 * t) + np.sin(2 * np.pi * 1320 * t)), SAMPLE_RATE)
    return [_write_tone(folder / "a440.wav", 440), _write_tone(folder / "a880.wav", 880), chord]


@pytest.fixture(scope="module")
def model(audios):
    return train_audio_codebook(audios, K, seed=0)


def test_mfcc_has_one_row_per_frame_and_n_mfcc_columns(tmp_path):
    # Estéreo a 44100 Hz: debe quedar mono a 22050 Hz, igual que un audio nativo.
    stereo = _write_tone(tmp_path / "stereo.wav", 440, sample_rate=44100, channels=2)
    native = _write_tone(tmp_path / "native.wav", 440)

    mfcc = extract_mfcc(stereo)

    assert mfcc.shape[1] == N_MFCC
    # Con hop de 512 muestras y padding centrado: 1 + 22050 // 512 = 44 frames.
    assert mfcc.shape == extract_mfcc(native).shape == (44, N_MFCC)


def test_different_audios_give_different_descriptors(audios):
    assert not np.allclose(extract_mfcc(audios[0]), extract_mfcc(audios[1]))


def test_codebook_has_k_centroids_and_histogram_has_k_bins(audios, model):
    centroids, idf = model
    descriptors = extract_mfcc(audios[0])

    histogram = count_words(assign_words(descriptors, centroids), K)

    assert centroids.shape == (K, N_MFCC)
    assert idf.shape == (K,)
    assert histogram.shape == (K,)
    assert histogram.sum() == len(descriptors)


def test_audio_vector_has_k_components_and_unit_norm(audios, model):
    centroids, idf = model

    vector = extract_audio_vector(audios[2], centroids, idf)

    assert len(vector) == K
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0)


def test_same_audio_gives_same_vector_and_zero_distance(audios, model):
    centroids, idf = model

    first = extract_audio_vector(audios[0], centroids, idf)
    second = extract_audio_vector(audios[0], centroids, idf)

    assert first == second
    assert euclidean_distance(first, second) == 0.0


def test_silence_and_empty_audio_give_finite_vectors(tmp_path, model):
    centroids, idf = model
    silence = tmp_path / "silence.wav"
    empty = tmp_path / "empty.wav"
    sf.write(silence, np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    sf.write(empty, np.zeros(0), SAMPLE_RATE)

    # El silencio sí tiene frames (todos iguales): vector normal de norma 1.
    silence_vector = extract_audio_vector(silence, centroids, idf)
    assert all(math.isfinite(x) for x in silence_vector)
    assert math.sqrt(sum(x * x for x in silence_vector)) == pytest.approx(1.0)

    # El audio vacío no tiene frames: cero descriptores y vector cero.
    assert extract_mfcc(empty).shape == (0, N_MFCC)
    assert extract_audio_vector(empty, centroids, idf) == [0.0] * K


def test_audio_vectors_round_trip_through_vectores_bin(tmp_path, audios, model):
    centroids, idf = model
    records = [
        VectorRecord(i, str(path), extract_audio_vector(path, centroids, idf))
        for i, path in enumerate(audios)
    ]
    file = tmp_path / "vectores.bin"
    write_vectors(file, records, dimension=K, count=len(records))

    read = list(iter_vectors(file))

    assert [(r.id, r.path) for r in read] == [(r.id, r.path) for r in records]
    assert [list(r.vector) for r in read] == [np.float32(r.vector).tolist() for r in records]


def test_sampled_codebook_keeps_shapes(audios):
    centroids, idf = train_audio_codebook(audios, K, seed=0, max_descriptors=60)

    assert centroids.shape == (K, N_MFCC)
    assert len(extract_audio_vector(audios[0], centroids, idf)) == K


def test_audio_training_extracts_each_file_twice_without_keeping_descriptors(monkeypatch, audios):
    calls = []
    refs = []

    def spy(path):
        gc.collect()
        # Al extraer un archivo, como mucho siguen vivos los descriptores del anterior.
        assert all(ref() is None for ref in refs[:-1])
        calls.append(path)
        descriptors = extract_mfcc(path)
        refs.append(weakref.ref(descriptors))
        return descriptors

    monkeypatch.setattr(audio_features, "extract_mfcc", spy)
    centroids, idf = audio_features.train_audio_codebook(audios, K, seed=0, max_descriptors=60)

    assert calls == [*audios, *audios]
    assert centroids.shape == (K, N_MFCC)
    assert idf.shape == (K,)
