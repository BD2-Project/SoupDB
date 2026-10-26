import gc
import math
import weakref

import cv2
import numpy as np
import pytest

from engine.indexes.vector.metrics import euclidean_distance
from engine.multimedia import image_features
from engine.multimedia.codebook import (
    assign_words,
    compute_idf,
    count_words,
    load_codebook,
    save_codebook,
    train_codebook,
)
from engine.multimedia.image_features import (
    extract_image_vector,
    extract_sift,
    train_image_codebook,
)
from engine.multimedia.vector_file import VectorRecord, iter_vectors, write_vectors

K = 8


def _textured_image(path, seed):
    # Ruido suavizado: tiene esquinas y manchas de sobra para que SIFT encuentre keypoints.
    noise = np.random.default_rng(seed).integers(0, 256, (128, 128), dtype=np.uint8)
    cv2.imwrite(str(path), cv2.GaussianBlur(noise, (5, 5), 0))
    return path


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    folder = tmp_path_factory.mktemp("images")
    return [_textured_image(folder / f"img{i}.png", seed=i) for i in range(3)]


@pytest.fixture(scope="module")
def model(images):
    return train_image_codebook(images, K, seed=0)


def test_sift_returns_128_dimensional_descriptors(images):
    descriptors = extract_sift(images[0])

    assert descriptors.ndim == 2
    assert len(descriptors) > 0
    assert descriptors.shape[1] == 128


def test_blank_image_has_no_descriptors_and_a_zero_vector(tmp_path, model):
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((64, 64), 127, dtype=np.uint8))
    centroids, idf = model

    assert extract_sift(blank).shape == (0, 128)
    assert extract_image_vector(blank, centroids, idf) == [0.0] * K


def test_codebook_has_k_centroids_and_is_reproducible(images):
    descriptors = extract_sift(images[0])

    centroids = train_codebook(descriptors, K, seed=0)

    assert centroids.shape == (K, 128)
    assert np.array_equal(centroids, train_codebook(descriptors, K, seed=0))


def test_codebook_round_trips_through_npz(tmp_path, model):
    centroids, idf = model
    save_codebook(tmp_path / "codebook.npz", centroids, idf)

    loaded_centroids, loaded_idf = load_codebook(tmp_path / "codebook.npz")

    assert np.array_equal(loaded_centroids, centroids)
    assert np.array_equal(loaded_idf, idf)


def test_histogram_has_k_bins_and_counts_every_descriptor(images, model):
    centroids, _ = model
    descriptors = extract_sift(images[0])

    histogram = count_words(assign_words(descriptors, centroids), K)

    assert histogram.shape == (K,)
    assert histogram.sum() == len(descriptors)


def test_count_words_example():
    assert count_words(np.array([0, 2, 2, 1, 2]), 4).tolist() == [1, 1, 3, 0]


def test_idf_formula():
    # 3 imágenes; la palabra 0 aparece en 2, la palabra 1 en 1.
    histograms = np.array([[1, 0], [2, 0], [0, 3]])

    idf = compute_idf(histograms)

    assert idf.shape == (2,)
    assert idf == pytest.approx([math.log(4 / 3) + 1, math.log(4 / 2) + 1])


def test_image_vector_has_k_components_and_unit_norm(images, model):
    centroids, idf = model

    vector = extract_image_vector(images[0], centroids, idf)

    assert idf.shape == (K,)
    assert len(vector) == K
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0)


def test_same_image_gives_same_vector_and_zero_distance(images, model):
    centroids, idf = model

    first = extract_image_vector(images[1], centroids, idf)
    second = extract_image_vector(images[1], centroids, idf)

    assert first == second
    assert euclidean_distance(first, second) == 0.0


def test_image_vectors_round_trip_through_vectores_bin(tmp_path, images, model):
    centroids, idf = model
    records = [
        VectorRecord(i, str(path), extract_image_vector(path, centroids, idf))
        for i, path in enumerate(images)
    ]
    file = tmp_path / "vectores.bin"
    write_vectors(file, records, dimension=K, count=len(records))

    read = list(iter_vectors(file))

    assert [(r.id, r.path) for r in read] == [(r.id, r.path) for r in records]
    assert [list(r.vector) for r in read] == [np.float32(r.vector).tolist() for r in records]


def test_sampled_codebook_keeps_shapes(images):
    centroids, idf = train_image_codebook(images, K, seed=0, max_descriptors=100)

    assert centroids.shape == (K, 128)
    assert len(extract_image_vector(images[0], centroids, idf)) == K


def test_sampled_codebook_computes_idf_over_every_image(tmp_path, images):
    # La imagen en blanco no aporta descriptores a la muestra, pero sí cuenta en N.
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((64, 64), 127, dtype=np.uint8))
    paths = [*images, blank]

    centroids, idf = train_image_codebook(paths, K, seed=0, max_descriptors=100)

    histograms = [count_words(assign_words(extract_sift(p), centroids), K) for p in paths]
    assert np.array_equal(idf, compute_idf(np.array(histograms)))


def test_image_training_extracts_each_file_twice_without_keeping_descriptors(monkeypatch, images):
    calls = []
    refs = []

    def spy(path):
        gc.collect()
        # Al extraer un archivo, como mucho siguen vivos los descriptores del anterior.
        assert all(ref() is None for ref in refs[:-1])
        calls.append(path)
        descriptors = extract_sift(path)
        refs.append(weakref.ref(descriptors))
        return descriptors

    monkeypatch.setattr(image_features, "extract_sift", spy)
    centroids, idf = image_features.train_image_codebook(images, K, seed=0, max_descriptors=100)

    assert calls == [*images, *images]
    assert centroids.shape == (K, 128)
    assert idf.shape == (K,)
