import numpy as np
import pytest

from engine.multimedia.codebook import sample_descriptors, train_codebook

DIM = 5


def _sets(*sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random((n, DIM), dtype=np.float32) for n in sizes]


def test_sample_is_capped_and_keeps_dimension():
    sets = _sets(40, 0, 60)

    sample = sample_descriptors(sets, max_descriptors=20, seed=0)

    assert sample.shape == (20, DIM)
    # Cada fila de la muestra es un descriptor original, no uno inventado.
    original = {row.tobytes() for row in np.vstack(sets)}
    assert all(row.tobytes() in original for row in sample)


def test_returns_everything_when_below_the_limit():
    sets = _sets(4, 6)

    assert np.array_equal(sample_descriptors(sets, max_descriptors=20), np.vstack(sets))


def test_same_seed_same_sample_and_different_seed_different_sample():
    sets = _sets(100)

    first = sample_descriptors(sets, max_descriptors=20, seed=1)

    assert np.array_equal(first, sample_descriptors(sets, max_descriptors=20, seed=1))
    assert not np.array_equal(first, sample_descriptors(sets, max_descriptors=20, seed=2))


def test_only_empty_sets_give_an_empty_sample_and_a_clear_training_error():
    sample = sample_descriptors(_sets(0, 0), max_descriptors=20)

    assert sample.shape == (0, DIM)
    with pytest.raises(ValueError, match="got 0"):
        train_codebook(sample, k=4)
