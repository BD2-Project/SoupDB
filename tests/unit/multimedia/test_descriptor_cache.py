import numpy as np
import pytest

from engine.multimedia.descriptor_cache import cache_path, cached, load_array, save_array


@pytest.mark.parametrize(
    "array",
    [
        np.random.default_rng(0).random((37, 128), dtype=np.float32),
        np.random.default_rng(1).random((5, 20)),
        np.empty((0, 128), dtype=np.float32),
    ],
    ids=["sift-float32", "mfcc-float64", "empty"],
)
def test_saved_array_is_loaded_exactly(tmp_path, array):
    file = tmp_path / "a" / "b" / "entry.npy"

    save_array(file, array)
    loaded = load_array(file)

    assert loaded.dtype == array.dtype
    assert loaded.shape == array.shape
    assert np.array_equal(loaded, array)
    # El temporal de la escritura atómica no queda en disco.
    assert [p.name for p in file.parent.iterdir()] == ["entry.npy"]


def test_missing_entry_loads_as_none(tmp_path):
    assert load_array(tmp_path / "nothing.npy") is None


def test_key_depends_on_kind_source_and_params(tmp_path):
    base = cache_path(tmp_path, "image", "img/a.png", {"max_side": 1024})

    assert base == cache_path(tmp_path, "image", "img/a.png", {"max_side": 1024})
    assert base != cache_path(tmp_path, "audio_segment", "img/a.png", {"max_side": 1024})
    assert base != cache_path(tmp_path, "image", "img/b.png", {"max_side": 1024})
    assert base != cache_path(tmp_path, "image", "img/a.png", {"max_side": 512})
    # El nombre es un hash fijo: nada de la ruta original (tildes, longitud) llega al disco.
    assert base.suffix == ".npy"
    assert len(base.stem) == 64
    assert base.parent.parent == tmp_path / "image"


def test_relative_and_absolute_paths_share_the_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    relative = cache_path("cache", "image", "fotos/ñandú.png", {})
    absolute = cache_path("cache", "image", tmp_path / "fotos" / "ñandú.png", {})

    assert relative == absolute


def test_cached_computes_only_on_miss(tmp_path):
    file = tmp_path / "entry.npy"
    calls = []

    def compute():
        calls.append(1)
        return np.arange(6, dtype=np.float32).reshape(2, 3)

    first = cached(file, compute)
    second = cached(file, compute)

    assert len(calls) == 1
    assert np.array_equal(first, second)
