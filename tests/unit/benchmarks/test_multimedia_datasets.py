"""Preparación de datasets multimedia con datos sintéticos (tamaños 2/4/6 por 1k/10k/100k)."""

import json
import math

import cv2
import numpy as np
import pytest
import soundfile as sf

from benchmarks.scripts import build_multimedia_vectors as wrapper
from benchmarks.scripts.build_multimedia_vectors import output_dir, validate_vector_file
from benchmarks.scripts.prepare_multimedia_datasets import (
    manifest_path,
    prepare_audio,
    prepare_images,
    size_label,
)
from engine.multimedia import build_vectors, image_features
from engine.multimedia.audio_features import SAMPLE_RATE, SEGMENT_SECONDS
from engine.multimedia.manifest import audio_rows
from engine.multimedia.vector_file import VectorRecord, iter_vectors, write_vectors

SIZES = (2, 4, 6)
K = 4


def _coco(root, n=7):
    root.mkdir(parents=True)
    # Nombres como los de COCO, creados en desorden para comprobar que se ordenan.
    for i in reversed(range(n)):
        noise = np.random.default_rng(i).integers(0, 256, (96, 96), dtype=np.uint8)
        cv2.imwrite(str(root / f"{i:012d}.jpg"), cv2.GaussianBlur(noise, (5, 5), 0))
    return root


def _fma(root, tracks=2, seconds=18.0):
    # Como FMA: carpetas 000/, 001/ con pistas NNNNNN.mp3 (aquí .wav). 18 s -> 4 segmentos.
    rng = np.random.default_rng(0)
    for i in range(tracks):
        path = root / f"{i:03d}" / f"{i:06d}.wav"
        path.parent.mkdir(parents=True)
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        signal = 0.4 * np.sin(2 * np.pi * 300 * (i + 1) * t) + 0.1 * rng.standard_normal(len(t))
        sf.write(path, signal, SAMPLE_RATE)
    return root


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def test_size_labels_match_the_requested_names(tmp_path):
    assert [size_label(n) for n in (1_000, 10_000, 100_000)] == ["1k", "10k", "100k"]
    assert manifest_path(tmp_path, "image", 1_000).name == "images_1k.jsonl"
    assert manifest_path(tmp_path, "audio", 100_000).name == "audio_100k.jsonl"


@pytest.mark.parametrize("kind", ["image", "audio"])
def test_manifests_are_nested_exact_prefixes(tmp_path, kind):
    if kind == "image":
        paths = prepare_images(_coco(tmp_path / "coco"), tmp_path / "m", SIZES)
    else:
        paths = prepare_audio(_fma(tmp_path / "fma"), tmp_path / "m", SIZES)
    small, medium, large = (_lines(p) for p in paths)

    assert [len(small), len(medium), len(large)] == list(SIZES)
    # 2 ⊂ 4 ⊂ 6, línea a línea: mismos IDs y mismas rutas.
    assert medium[: len(small)] == small
    assert large[: len(medium)] == medium
    assert [json.loads(line)["id"] for line in large] == list(range(6))


def test_same_files_give_the_same_manifests(tmp_path):
    coco = _coco(tmp_path / "coco")

    first = prepare_images(coco, tmp_path / "a", SIZES)
    second = prepare_images(coco, tmp_path / "b", SIZES)

    assert [p.read_bytes() for p in first] == [p.read_bytes() for p in second]
    assert [json.loads(line)["path"] for line in _lines(first[0])] == [
        "000000000000.jpg",
        "000000000001.jpg",
    ]


def test_too_few_objects_stop_without_leaving_manifests(tmp_path):
    with pytest.raises(SystemExit, match="se necesitan 6"):
        prepare_images(_coco(tmp_path / "coco", n=5), tmp_path / "m", SIZES)
    # 2 pistas de 18 s = 8 segmentos < 10.
    with pytest.raises(SystemExit, match="solo hay 8"):
        prepare_audio(_fma(tmp_path / "fma"), tmp_path / "m", (2, 10))
    assert list((tmp_path / "m").glob("*")) == []


def test_missing_dataset_folder_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="no existe"):
        prepare_images(tmp_path / "nada", tmp_path / "m", SIZES)


def test_unreadable_audio_is_skipped_only_when_asked(tmp_path):
    fma = _fma(tmp_path / "fma")
    (fma / "000" / "000000x.wav").write_bytes(b"not audio")

    with pytest.raises(Exception):  # noqa: B017 - el tipo depende del decodificador
        list(audio_rows(fma))
    rows = list(audio_rows(fma, skip_unreadable=True))

    assert {r["path"] for r in rows} == {"000/000000.wav", "001/000001.wav"}
    assert [r["id"] for r in rows] == list(range(8))


# --- validación de vectores.bin ---------------------------------------------------------


@pytest.fixture
def manifest(tmp_path):
    rows = [
        {"id": 0, "type": "audio_segment", "path": "000/000002.mp3", "segment": 0},
        {"id": 1, "type": "audio_segment", "path": "000/000002.mp3", "segment": 1},
        {"id": 2, "type": "audio_segment", "path": "000/000005.mp3", "segment": 0},
    ]
    path = tmp_path / "audio_3.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _records(**changes):
    records = [
        VectorRecord(0, "000/000002.mp3#segment=000000", [0.5, 0.5, 0.5, 0.5]),
        VectorRecord(1, "000/000002.mp3#segment=000001", [1.0, 0.0, 0.0, 0.0]),
        VectorRecord(2, "000/000005.mp3#segment=000000", [0.0, 1.0, 0.0, 0.0]),
    ]
    for index, record in changes.items():
        records[int(index.removeprefix("r"))] = record
    return records


def _write(tmp_path, records, dimension=K):
    path = tmp_path / "vectores.bin"
    write_vectors(path, records, dimension=dimension, count=len(records))
    return path


def test_valid_vector_file_passes(tmp_path, manifest):
    assert validate_vector_file(_write(tmp_path, _records()), manifest, K) == 3


def test_validation_detects_wrong_count(tmp_path, manifest):
    with pytest.raises(ValueError, match="count mismatch"):
        validate_vector_file(_write(tmp_path, _records()[:2]), manifest, K)
    extra = [*_records(), VectorRecord(3, "x#segment=000000", [0.0] * K)]
    with pytest.raises(ValueError, match="count mismatch"):
        validate_vector_file(_write(tmp_path, extra), manifest, K)


def test_validation_detects_wrong_dimension(tmp_path, manifest):
    records = [VectorRecord(r.id, r.path, [*r.vector, 0.0]) for r in _records()]

    with pytest.raises(ValueError, match="dimension 5, expected 4"):
        validate_vector_file(_write(tmp_path, records, dimension=5), manifest, K)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_validation_detects_nan_and_infinity(tmp_path, manifest, bad):
    records = _records(r1=VectorRecord(1, "000/000002.mp3#segment=000001", [bad, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_vector_file(_write(tmp_path, records), manifest, K)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (VectorRecord(0, "000/000002.mp3#segment=000001", [1.0] * K), "duplicated id 0"),
        (VectorRecord(7, "000/000002.mp3#segment=000001", [1.0] * K), "id 7, expected 1"),
        (VectorRecord(1, "000/000002.mp3", [1.0] * K), "path"),
    ],
    ids=["duplicated-id", "unexpected-id", "wrong-path"],
)
def test_validation_detects_wrong_ids_and_paths(tmp_path, manifest, record, message):
    with pytest.raises(ValueError, match=message):
        validate_vector_file(_write(tmp_path, _records(r1=record)), manifest, K)


# --- wrapper sobre build_vectors ------------------------------------------------------


def _run_wrapper(kind, root, data_dir, sizes):
    args = [kind, "--root", str(root), "--data-dir", str(data_dir), "--sizes"]
    wrapper.main([*args, *map(str, sizes), "--k", str(K), "--max-descriptors", "200"])


def test_wrapper_builds_every_size_with_one_extraction_per_image(monkeypatch, tmp_path):
    coco = _coco(tmp_path / "coco")
    data = tmp_path / "data"
    prepare_images(coco, data / "manifests", SIZES)

    sift_calls, build_calls = [], []
    real_sift, real_build = image_features.extract_sift, build_vectors.build

    def sift_spy(path, max_side=image_features.MAX_IMAGE_SIDE):
        sift_calls.append(path)
        return real_sift(path, max_side)

    def build_spy(*args):
        build_calls.append(args)
        return real_build(*args)

    monkeypatch.setattr(image_features, "extract_sift", sift_spy)
    monkeypatch.setattr(build_vectors, "build", build_spy)
    _run_wrapper("image", coco, data, SIZES)

    # Tres tamaños anidados con caché compartida: SIFT una vez por imagen (6), no 2+4+6.
    assert len(build_calls) == 3
    assert len(sift_calls) == len(set(sift_calls)) == 6
    for size in SIZES:
        out = output_dir(data, "image", size)
        records = list(iter_vectors(out / "vectores.bin"))
        assert [r.id for r in records] == list(range(size))
        assert (out / "codebook.npz").is_file()
        params = json.loads((out / "params.json").read_text(encoding="utf-8"))
        assert params["size"] == size
        assert params["k"] == K
        assert params["seed"] == wrapper.SEED
        assert params["image_max_side"] == 1024

    # Repetir el comando (p. ej. tras un corte) no vuelve a extraer nada.
    _run_wrapper("image", coco, data, SIZES)
    assert len(sift_calls) == 6


def test_wrapper_builds_audio_segments(tmp_path):
    fma = _fma(tmp_path / "fma")
    data = tmp_path / "data"
    prepare_audio(fma, data / "manifests", (3, 8))

    _run_wrapper("audio", fma, data, (3, 8))

    records = list(iter_vectors(output_dir(data, "audio", 8) / "vectores.bin"))
    assert [r.path for r in records[:5]] == [
        "000/000000.wav#segment=000000",
        "000/000000.wav#segment=000001",
        "000/000000.wav#segment=000002",
        "000/000000.wav#segment=000003",
        "001/000001.wav#segment=000000",
    ]
    assert all(len(r.vector) == K for r in records)


def test_wrapper_requires_the_manifest(tmp_path):
    with pytest.raises(SystemExit, match="prepare_multimedia_datasets"):
        _run_wrapper("image", tmp_path, tmp_path / "data", (2,))


def test_parameters_are_the_documented_ones():
    assert (wrapper.K, wrapper.MAX_DESCRIPTORS, wrapper.SEED) == (128, 50_000, 20261024)
    assert image_features.MAX_IMAGE_SIDE == 1024
    assert SEGMENT_SECONDS == 5.0
