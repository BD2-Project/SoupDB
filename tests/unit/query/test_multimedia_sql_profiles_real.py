"""C15: real codebook/TF-IDF SQL vectorizer, with deterministic descriptors."""

import numpy as np
import pytest

from engine.common.errors import QueryExecutionError
from engine.multimedia.codebook import save_codebook
from engine.multimedia.sql_profiles import MultimediaProfile, MultimediaQueryVectorizer


def _image_profile(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"placeholder")
    (root / "b.jpg").write_bytes(b"placeholder")
    centroids = np.zeros((2, 128), dtype=np.float32)
    centroids[1, 0] = 10
    codebook = tmp_path / "image_codebook.npz"
    save_codebook(codebook, centroids, np.ones(2))
    return MultimediaProfile("image-v1", "image", root, codebook)


def test_image_profile_uses_original_bow_tfidf_pipeline(tmp_path, monkeypatch) -> None:
    profile = _image_profile(tmp_path)
    from engine.multimedia import image_features

    def descriptors(path):
        out = np.zeros((1, 128), dtype=np.float32)
        if path.stem == "b":
            out[0, 0] = 10
        return out

    monkeypatch.setattr(image_features, "extract_sift", descriptors)
    resolver = MultimediaQueryVectorizer({"image-v1": profile})
    assert resolver.dimension("image-v1") == 2
    assert resolver.vectorize("a.jpg", profile_id="image-v1") == (1.0, 0.0)
    assert resolver.vectorize("b.jpg", profile_id="image-v1") == (0.0, 1.0)


def test_image_profile_never_reads_outside_data_root(tmp_path) -> None:
    profile = _image_profile(tmp_path)
    resolver = MultimediaQueryVectorizer({"image-v1": profile})
    with pytest.raises(QueryExecutionError, match="outside"):
        resolver.vectorize("../image_codebook.npz", profile_id="image-v1")
    with pytest.raises(QueryExecutionError, match="relative"):
        resolver.vectorize(str(profile.root / "a.jpg"), profile_id="image-v1")
    with pytest.raises(QueryExecutionError, match="unknown multimedia profile"):
        resolver.vectorize("a.jpg", profile_id="nonexistent")


def test_profile_dimension_and_bad_codebook(tmp_path) -> None:
    profile = _image_profile(tmp_path)
    wrong = tmp_path / "invalid.npz"
    save_codebook(wrong, np.zeros((3, 20)), np.ones(3))
    resolver = MultimediaQueryVectorizer(
        {"broken": MultimediaProfile("broken", "image", profile.root, wrong)}
    )
    with pytest.raises(QueryExecutionError, match="invalid codebook"):
        resolver.dimension("broken")


def test_audio_segment_profile_uses_mfcc_bow(tmp_path, monkeypatch) -> None:
    from engine.multimedia import audio_features

    root = tmp_path / "audio"
    root.mkdir()
    (root / "track.wav").write_bytes(b"placeholder")
    centroids = np.zeros((2, 20), dtype=np.float32)
    centroids[1, 0] = 10
    codebook = tmp_path / "audio_codebook.npz"
    save_codebook(codebook, centroids, np.ones(2))

    def iter_segments(path):
        assert path.name == "track.wav"
        yield 0, np.array([0.0])
        yield 1, np.array([10.0])

    def mfcc(signal):
        vector = np.zeros((1, 20), dtype=np.float32)
        vector[0, 0] = signal[0]
        return vector

    monkeypatch.setattr(audio_features, "iter_audio_segments", iter_segments)
    monkeypatch.setattr(audio_features, "mfcc_from_signal", mfcc)
    resolver = MultimediaQueryVectorizer(
        {"audio-v1": MultimediaProfile("audio-v1", "audio_segment", root, codebook)}
    )
    assert resolver.vectorize("track.wav", profile_id="audio-v1") == (1.0, 0.0)
    assert resolver.vectorize("track.wav#segment=000001", profile_id="audio-v1") == (0.0, 1.0)
    with pytest.raises(QueryExecutionError, match="not found"):
        resolver.vectorize("track.wav#segment=000002", profile_id="audio-v1")


def test_real_sift_descriptor_and_codebook_query(tmp_path) -> None:
    """Smoke test: decode an actual PNG and run real SIFT + BoW TF-IDF."""
    import cv2

    from engine.multimedia.image_features import extract_sift

    root = tmp_path / "photos"
    root.mkdir()
    image = np.random.default_rng(42).integers(0, 256, size=(128, 128), dtype=np.uint8)
    assert cv2.imwrite(str(root / "texture.png"), image)
    desc = extract_sift(root / "texture.png")
    assert desc.shape[1] == 128
    assert len(desc) > 1

    codebook = tmp_path / "real_sift.npz"
    save_codebook(codebook, desc[:2], np.ones(2))
    resolver = MultimediaQueryVectorizer(
        {"real": MultimediaProfile("real", "image", root, codebook)}
    )
    values = resolver.vectorize("texture.png", profile_id="real")
    assert len(values) == 2
    assert all(np.isfinite(values))
    assert np.linalg.norm(values) == pytest.approx(1.0)
