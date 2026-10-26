import json
import math
import re

import cv2
import numpy as np
import pytest
import soundfile as sf

from engine.indexes.vector.metrics import euclidean_distance
from engine.multimedia import audio_features, build_vectors, image_features
from engine.multimedia.audio_features import SAMPLE_RATE
from engine.multimedia.build_vectors import (
    build,
    build_vector_file,
    descriptor_for_row,
    main,
    train_from_manifest,
    validate_manifest,
)
from engine.multimedia.codebook import load_codebook
from engine.multimedia.manifest import audio_rows, image_rows, read_manifest, write_manifest
from engine.multimedia.vector_file import iter_vectors

K = 8


def _image_root(root):
    for i, name in enumerate(
        ["gatos/1.png", "gatos/2.png", "perros/1.png", "perros/2.png", "x.png"]
    ):
        noise = np.random.default_rng(i).integers(0, 256, (128, 128), dtype=np.uint8)
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / name), cv2.GaussianBlur(noise, (5, 5), 0))
    return root


def _audio_root(root):
    # 12 s -> 3 segmentos, 7 s -> 2 segmentos. El ruido da frames variados para K-Means.
    rng = np.random.default_rng(0)
    for name, frequency, seconds in [("jazz/a.wav", 330, 12.0), ("rock/b.wav", 880, 7.0)]:
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        signal = 0.4 * np.sin(2 * np.pi * frequency * t) + 0.1 * rng.standard_normal(len(t))
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        sf.write(root / name, signal, SAMPLE_RATE)
    return root


@pytest.fixture
def image_dataset(tmp_path):
    root = _image_root(tmp_path / "imgs")
    manifest = tmp_path / "images.jsonl"
    write_manifest(manifest, image_rows(root))
    return manifest, root


@pytest.fixture
def audio_dataset(tmp_path):
    root = _audio_root(tmp_path / "audio")
    manifest = tmp_path / "audio.jsonl"
    write_manifest(manifest, audio_rows(root))
    return manifest, root


def _build(tmp_path, dataset, name="run", cache="cache"):
    manifest, root = dataset
    codebook = tmp_path / f"{name}.npz"
    vectors = tmp_path / f"{name}.bin"
    build(manifest, root, tmp_path / cache, codebook, vectors, K, max_descriptors=300, seed=42)
    return codebook, vectors


def _check_vector_file(vectors, manifest):
    rows = list(read_manifest(manifest))
    records = list(iter_vectors(vectors))

    assert len(records) == len(rows)
    assert [r.id for r in records] == [row["id"] for row in rows]
    for record in records:
        assert len(record.vector) == K
        assert all(math.isfinite(x) for x in record.vector)
        assert math.sqrt(sum(x * x for x in record.vector)) == pytest.approx(1.0, abs=1e-6)
        assert euclidean_distance(record.vector, record.vector) == 0.0
    return rows, records


def test_image_manifest_gives_codebook_and_vector_file(tmp_path, image_dataset):
    codebook, vectors = _build(tmp_path, image_dataset)

    centroids, idf = load_codebook(codebook)
    rows, records = _check_vector_file(vectors, image_dataset[0])

    assert centroids.shape == (K, 128)
    assert idf.shape == (K,)
    assert len(records) == 5
    assert [r.path for r in records] == [row["path"] for row in rows]


def test_audio_manifest_gives_codebook_and_vector_file(tmp_path, audio_dataset):
    codebook, vectors = _build(tmp_path, audio_dataset)

    centroids, idf = load_codebook(codebook)
    rows, records = _check_vector_file(vectors, audio_dataset[0])

    assert centroids.shape == (K, 20)
    assert idf.shape == (K,)
    assert [r.path for r in records] == [
        "jazz/a.wav#segment=000000",
        "jazz/a.wav#segment=000001",
        "jazz/a.wav#segment=000002",
        "rock/b.wav#segment=000000",
        "rock/b.wav#segment=000001",
    ]
    assert all(re.fullmatch(r".+#segment=\d{6}", r.path) for r in records)


def test_audio_rows_use_their_own_segment_not_the_whole_track(tmp_path, audio_dataset):
    manifest, root = audio_dataset
    rows = list(read_manifest(manifest))
    segments = [s for _, s in audio_features.iter_audio_segments(root / "jazz/a.wav")]

    for row, samples in zip(rows[:3], segments, strict=True):
        descriptor = descriptor_for_row(row, root, tmp_path / "cache")
        assert np.array_equal(descriptor, audio_features.mfcc_from_signal(samples))


def test_same_seed_and_files_give_the_same_vectors(tmp_path, image_dataset):
    # Cachés distintas: la segunda ejecución vuelve a extraer todo desde cero.
    _, first = _build(tmp_path, image_dataset, name="a", cache="cache_a")
    _, second = _build(tmp_path, image_dataset, name="b", cache="cache_b")

    assert first.read_bytes() == second.read_bytes()


def test_saved_codebook_reproduces_the_same_vectors(tmp_path, audio_dataset):
    manifest, root = audio_dataset
    codebook, vectors = _build(tmp_path, audio_dataset)
    centroids, idf = load_codebook(codebook)

    build_vector_file(manifest, root, tmp_path / "cache", centroids, idf, tmp_path / "again.bin")

    assert (tmp_path / "again.bin").read_bytes() == vectors.read_bytes()


def test_descriptors_are_extracted_once_and_reused_by_a_second_run(
    monkeypatch, tmp_path, image_dataset
):
    calls = []
    real = image_features.extract_sift

    def spy(path, max_side=image_features.MAX_IMAGE_SIDE):
        calls.append(path)
        return real(path, max_side)

    monkeypatch.setattr(image_features, "extract_sift", spy)

    # 1ª ejecución: tres pasadas sobre el manifest, pero SIFT una vez por imagen.
    _, first = _build(tmp_path, image_dataset, name="a")
    assert len(calls) == 5

    # 2ª ejecución con la misma caché: nada de SIFT y mismo resultado.
    _, second = _build(tmp_path, image_dataset, name="b")
    assert len(calls) == 5
    assert first.read_bytes() == second.read_bytes()


def test_second_audio_run_does_not_decode_tracks(monkeypatch, tmp_path, audio_dataset):
    _build(tmp_path, audio_dataset, name="a")
    loads = []
    real = audio_features.load_audio
    monkeypatch.setattr(audio_features, "load_audio", lambda p: loads.append(p) or real(p))

    _build(tmp_path, audio_dataset, name="b")

    assert loads == []


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_mixed_manifest_is_rejected(tmp_path):
    manifest = _write_jsonl(
        tmp_path / "mixed.jsonl",
        [
            {"id": 0, "type": "image", "path": "a.png"},
            {"id": 1, "type": "audio_segment", "path": "a.wav", "segment": 0},
        ],
    )

    with pytest.raises(ValueError, match="mixed manifest"):
        validate_manifest(manifest)
    with pytest.raises(ValueError, match="mixed manifest"):
        train_from_manifest(manifest, tmp_path, tmp_path / "cache", K)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": 0, "type": "audio_segment", "path": "a.wav"}], "missing fields"),
        ([{"id": 0, "type": "image"}], "missing fields"),
        ([{"id": 0, "type": "video", "path": "a.mp4"}], "unknown type"),
        ([], "empty manifest"),
    ],
    ids=["no-segment", "no-path", "unknown-type", "empty"],
)
def test_invalid_manifest_is_rejected(tmp_path, rows, message):
    manifest = _write_jsonl(tmp_path / "bad.jsonl", rows)

    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest)


def test_vector_records_are_streamed_to_write_vectors(monkeypatch, tmp_path, image_dataset):
    manifest, root = image_dataset
    centroids, idf = train_from_manifest(manifest, root, tmp_path / "cache", K, 300, seed=42)
    calls = []
    real_descriptor = build_vectors.descriptor_for_row
    real_write = build_vectors.write_vectors

    def descriptor_spy(row, root, cache_dir):
        calls.append(row["id"])
        return real_descriptor(row, root, cache_dir)

    def write_spy(path, records, *, dimension, count):
        # Llega un generador, no una lista; y cada vector se calcula al pedirlo.
        assert iter(records) is records
        assert calls == []

        def checked():
            for i, record in enumerate(records):
                assert calls == list(range(i + 1))
                yield record

        real_write(path, checked(), dimension=dimension, count=count)

    monkeypatch.setattr(build_vectors, "descriptor_for_row", descriptor_spy)
    monkeypatch.setattr(build_vectors, "write_vectors", write_spy)
    count = build_vector_file(
        manifest, root, tmp_path / "cache", centroids, idf, tmp_path / "v.bin"
    )

    assert count == 5
    assert calls == [0, 1, 2, 3, 4]


def test_command_line_writes_codebook_and_vectors(tmp_path, image_dataset, capsys):
    manifest, root = image_dataset
    codebook, vectors = tmp_path / "codebook.npz", tmp_path / "vectores.bin"

    main(
        [str(manifest), str(root), str(tmp_path / "cache"), str(codebook), str(vectors)]
        + ["--k", str(K), "--max-descriptors", "300", "--seed", "42"]
    )

    assert load_codebook(codebook)[0].shape == (K, 128)
    _check_vector_file(vectors, manifest)
    assert "5 vectores" in capsys.readouterr().out
