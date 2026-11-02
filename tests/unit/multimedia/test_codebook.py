import gc
import weakref

import numpy as np
import pytest

from engine.multimedia.codebook import sample_descriptors, train_codebook

DIM = 5


def _sets(*sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random((n, DIM), dtype=np.float32) for n in sizes]


def test_sample_reads_blocks_one_at_a_time_and_is_capped():
    sizes = (40, 0, 60, 30)
    refs = []
    released = []

    def blocks():
        # Generador: cada bloque se crea al pedirlo, como un archivo recién extraído.
        for size in sizes:
            gc.collect()
            # Al pedir un bloque nuevo, como mucho sigue vivo el anterior.
            released.append(all(ref() is None for ref in refs[:-1]))
            block = _sets(size, seed=size)[0]
            refs.append(weakref.ref(block))
            yield block

    sample = sample_descriptors(blocks(), max_descriptors=20, seed=0)

    assert all(released)
    assert sample.shape == (20, DIM)
    # Cada fila de la muestra es un descriptor original, no uno inventado.
    original = {row.tobytes() for size in sizes for row in _sets(size, seed=size)[0]}
    assert all(row.tobytes() in original for row in sample)


def test_every_descriptor_has_the_same_chance():
    # 50 descriptores en 5 bloques y muestra de 10: cada uno debe salir ~20 % de las veces.
    blocks = [np.arange(i, i + 10, dtype=np.float32).reshape(10, 1) for i in range(0, 50, 10)]
    counts = np.zeros(50)

    for seed in range(2000):
        chosen = sample_descriptors(blocks, max_descriptors=10, seed=seed).ravel().astype(int)
        assert len(set(chosen)) == 10
        counts[chosen] += 1

    # Esperado 2000 · 10/50 = 400 por descriptor; ±100 es más de 5 desviaciones típicas.
    assert counts.tolist() == pytest.approx([400] * 50, abs=100)


def test_returns_everything_when_below_the_limit_or_without_limit():
    sets = _sets(4, 6)

    assert np.array_equal(sample_descriptors(sets, max_descriptors=20), np.vstack(sets))
    assert np.array_equal(sample_descriptors(iter(sets), max_descriptors=None), np.vstack(sets))


def test_same_seed_same_sample_and_different_seed_different_sample():
    sets = _sets(100)

    first = sample_descriptors(sets, max_descriptors=20, seed=1)

    assert np.array_equal(first, sample_descriptors(sets, max_descriptors=20, seed=1))
    assert not np.array_equal(first, sample_descriptors(sets, max_descriptors=20, seed=2))


def test_only_empty_sets_give_an_empty_sample_and_a_clear_training_error():
    sample = sample_descriptors(_sets(0, 0), max_descriptors=20)

    assert sample.shape == (0, DIM)
    assert len(sample_descriptors([], max_descriptors=20)) == 0
    with pytest.raises(ValueError, match="got 0"):
        train_codebook(sample, k=4)
