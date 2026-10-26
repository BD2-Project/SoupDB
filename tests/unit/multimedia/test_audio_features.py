import gc
import math
import warnings
import weakref

import numpy as np
import pytest
import soundfile as sf

from engine.indexes.vector.metrics import euclidean_distance
from engine.multimedia import audio_features
from engine.multimedia.audio_features import (
    MIN_SEGMENT_RATIO,
    N_MFCC,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    count_audio_segments,
    extract_audio_vector,
    extract_mfcc,
    extract_mfcc_cached,
    iter_audio_segments,
    load_audio,
    mfcc_from_signal,
    segment_id,
    train_audio_codebook,
)
from engine.multimedia.codebook import assign_words, count_words
from engine.multimedia.vector_file import VectorRecord, iter_vectors, write_vectors

K = 8


class SimulatedCrash(Exception):
    """Corte simulado de una ejecución larga (como un Ctrl+C a mitad)."""


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


@pytest.fixture
def track_18s(tmp_path):
    return _write_tone(tmp_path / "track.wav", 440, seconds=18.0)


def test_18_second_track_gives_four_segments_of_5_seconds(track_18s):
    segments = list(iter_audio_segments(track_18s, duration=5.0))

    assert [index for index, _ in segments] == [0, 1, 2, 3]
    # 0-5 s, 5-10 s, 10-15 s y 15-18 s: el último, más corto, no se descarta.
    assert [len(samples) / SAMPLE_RATE for _, samples in segments] == [5.0, 5.0, 5.0, 3.0]
    assert np.array_equal(np.concatenate([s for _, s in segments]), load_audio(track_18s))
    assert count_audio_segments(track_18s, duration=5.0) == 4


def test_segment_ids_are_stable():
    assert segment_id("rock/canción.mp3", 0) == "rock/canción.mp3#segment=000000"
    assert segment_id("rock/canción.mp3", 1) == "rock/canción.mp3#segment=000001"
    assert segment_id("rock/canción.mp3", 1) == segment_id("rock/canción.mp3", 1)


def test_full_file_mfcc_uses_the_same_formula_as_segments(track_18s):
    assert np.array_equal(extract_mfcc(track_18s), mfcc_from_signal(load_audio(track_18s)))


def test_every_segment_mfcc_has_n_mfcc_columns_and_no_nan(tmp_path, track_18s):
    for index, samples in iter_audio_segments(track_18s):
        cached = extract_mfcc_cached(track_18s, index, tmp_path / "cache")

        assert cached.shape[1] == N_MFCC
        assert len(cached) > 0
        assert np.isfinite(cached).all()
        assert np.array_equal(cached, mfcc_from_signal(samples))


def _spy_audio(monkeypatch):
    loads, mfccs = [], []
    real_load, real_mfcc = audio_features.load_audio, audio_features.mfcc_from_signal

    def load(path):
        loads.append(path)
        return real_load(path)

    def mfcc(signal, n_mfcc=N_MFCC):
        mfccs.append(len(signal))
        return real_mfcc(signal, n_mfcc)

    monkeypatch.setattr(audio_features, "load_audio", load)
    monkeypatch.setattr(audio_features, "mfcc_from_signal", mfcc)
    return loads, mfccs


def test_cached_mfcc_is_not_recomputed(monkeypatch, tmp_path, track_18s):
    cache = tmp_path / "cache"
    loads, mfccs = _spy_audio(monkeypatch)

    first = extract_mfcc_cached(track_18s, 0, cache)
    # El primer fallo decodifica la pista una vez y guarda sus 4 segmentos.
    assert loads == [track_18s]
    assert len(mfccs) == 4

    again = extract_mfcc_cached(track_18s, 0, cache)
    others = [extract_mfcc_cached(track_18s, i, cache) for i in (1, 2, 3)]

    assert loads == [track_18s]
    assert len(mfccs) == 4
    assert np.array_equal(first, again)
    assert all(m.shape[1] == N_MFCC for m in others)


def test_segment_out_of_range_raises(tmp_path, track_18s):
    with pytest.raises(IndexError):
        extract_mfcc_cached(track_18s, 4, tmp_path / "cache")


def test_interrupted_audio_run_resumes_from_cache(monkeypatch, tmp_path):
    tracks = [_write_tone(tmp_path / f"t{i}.wav", 220 * (i + 1), seconds=13.0) for i in range(3)]
    work = [(track, segment) for track in tracks for segment in range(3)]
    cache = tmp_path / "cache"

    real_load = audio_features.load_audio
    loaded = []

    def crashes_on_second_track(path):
        if path == tracks[1]:
            raise SimulatedCrash
        loaded.append(path)
        return real_load(path)

    monkeypatch.setattr(audio_features, "load_audio", crashes_on_second_track)
    with pytest.raises(SimulatedCrash):
        for track, segment in work:
            extract_mfcc_cached(track, segment, cache)
    assert loaded == [tracks[0]]

    # 2ª ejecución: la 1ª pista sale entera de la caché; solo se decodifican las demás.
    monkeypatch.setattr(audio_features, "load_audio", real_load)
    loads, _ = _spy_audio(monkeypatch)
    results = [extract_mfcc_cached(track, segment, cache) for track, segment in work]

    assert loads == tracks[1:]
    expected = [mfcc_from_signal(s) for t in tracks for _, s in iter_audio_segments(t)]
    assert all(np.array_equal(r, e) for r, e in zip(results, expected, strict=True))


SEGMENT = round(SEGMENT_SECONDS * SAMPLE_RATE)  # 110250 muestras


def _write_samples(path, n):
    # Exactamente n muestras a 22050 Hz: librosa no remuestrea y la longitud no cambia.
    rng = np.random.default_rng(n)
    sf.write(path, 0.3 * rng.standard_normal(n), SAMPLE_RATE)
    return path


def _lengths(path):
    return [len(samples) for _, samples in iter_audio_segments(path)]


def test_tiny_decoding_tail_is_not_a_segment(tmp_path):
    # 15 s + 60 muestras: la cola de 60 muestras no se convierte en objeto.
    track = _write_samples(tmp_path / "tail.wav", 3 * SEGMENT + 60)

    assert _lengths(track) == [SEGMENT] * 3
    assert count_audio_segments(track) == 3
    with pytest.raises(IndexError):
        extract_mfcc_cached(track, 3, tmp_path / "cache")
    # Ya no hay segmentos más cortos que n_fft: librosa no tiene nada de qué avisar.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for index in range(3):
            assert extract_mfcc_cached(track, index, tmp_path / "cache").shape[1] == N_MFCC


def test_tail_of_a_tenth_of_a_second_is_dropped(tmp_path):
    track = _write_tone(tmp_path / "t.wav", 440, seconds=15.1)

    assert _lengths(track) == [SEGMENT] * 3


def test_last_segment_of_exactly_half_is_kept(tmp_path):
    half = round(MIN_SEGMENT_RATIO * SEGMENT)  # 2,5 s
    kept = _write_samples(tmp_path / "kept.wav", 2 * SEGMENT + half)
    dropped = _write_samples(tmp_path / "dropped.wav", 2 * SEGMENT + half - 1)

    assert _lengths(kept) == [SEGMENT, SEGMENT, half]
    assert _lengths(dropped) == [SEGMENT, SEGMENT]


@pytest.mark.parametrize(
    "n",
    [0, 60, SEGMENT // 2 - 1, SEGMENT // 2 + 1, SEGMENT, 3 * SEGMENT + 60, 3 * SEGMENT + 66150],
    ids=["empty", "60", "under-half", "over-half", "one", "tail-60", "18s"],
)
def test_count_matches_iteration(tmp_path, n):
    track = _write_samples(tmp_path / "t.wav", n)

    lengths = _lengths(track)

    assert count_audio_segments(track) == len(lengths)
    # Todos completos menos, como mucho, el último, que dura al menos la mitad.
    assert all(length == SEGMENT for length in lengths[:-1])
    assert all(length >= MIN_SEGMENT_RATIO * SEGMENT for length in lengths)
