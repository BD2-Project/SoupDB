import cv2
import numpy as np
import soundfile as sf

from engine.multimedia.audio_features import SAMPLE_RATE, count_audio_segments, iter_audio_segments
from engine.multimedia.manifest import (
    audio_rows,
    image_rows,
    list_media_files,
    main,
    read_manifest,
    write_manifest,
)


def _image_dir(root):
    for name in ["b/2.png", "a/10.jpg", "a/9.jpg", "c.PNG"]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / name), np.zeros((8, 8), dtype=np.uint8))
    # Lo que no es imagen no entra en el manifest.
    (root / "notas.txt").write_text("x")
    return root


def _audio_dir(root):
    for name, seconds in [("rock/b.wav", 8.0), ("jazz/a.wav", 18.0)]:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        sf.write(root / name, np.zeros(int(seconds * SAMPLE_RATE)), SAMPLE_RATE)
    return root


def test_files_are_sorted_by_relative_path(tmp_path):
    files = list_media_files(_image_dir(tmp_path), {".png", ".jpg"})

    assert files == ["a/10.jpg", "a/9.jpg", "b/2.png", "c.PNG"]


def test_image_manifest_is_identical_when_built_twice(tmp_path):
    root = _image_dir(tmp_path / "imgs")

    write_manifest(tmp_path / "m1.jsonl", image_rows(root))
    write_manifest(tmp_path / "m2.jsonl", image_rows(root))

    assert (tmp_path / "m1.jsonl").read_bytes() == (tmp_path / "m2.jsonl").read_bytes()
    assert list(read_manifest(tmp_path / "m1.jsonl")) == [
        {"id": 0, "type": "image", "path": "a/10.jpg"},
        {"id": 1, "type": "image", "path": "a/9.jpg"},
        {"id": 2, "type": "image", "path": "b/2.png"},
        {"id": 3, "type": "image", "path": "c.PNG"},
    ]


def test_audio_manifest_has_one_row_per_segment(tmp_path):
    root = _audio_dir(tmp_path / "audio")

    write_manifest(tmp_path / "m1.jsonl", audio_rows(root))
    write_manifest(tmp_path / "m2.jsonl", audio_rows(root))
    rows = list(read_manifest(tmp_path / "m1.jsonl"))

    assert (tmp_path / "m1.jsonl").read_bytes() == (tmp_path / "m2.jsonl").read_bytes()
    assert [(r["id"], r["path"], r["segment"], r["start"], r["end"]) for r in rows] == [
        (0, "jazz/a.wav", 0, 0.0, 5.0),
        (1, "jazz/a.wav", 1, 5.0, 10.0),
        (2, "jazz/a.wav", 2, 10.0, 15.0),
        (3, "jazz/a.wav", 3, 15.0, 18.0),
        (4, "rock/b.wav", 0, 0.0, 5.0),
        (5, "rock/b.wav", 1, 5.0, 8.0),
    ]
    assert {r["type"] for r in rows} == {"audio_segment"}


def test_limit_keeps_the_first_objects(tmp_path):
    root = _image_dir(tmp_path / "imgs")

    count = write_manifest(tmp_path / "m.jsonl", image_rows(root), limit=2)

    assert count == 2
    assert [r["id"] for r in read_manifest(tmp_path / "m.jsonl")] == [0, 1]


def test_command_line_builds_the_manifest(tmp_path, capsys):
    root = _audio_dir(tmp_path / "audio")

    main(["audio", str(root), str(tmp_path / "m.jsonl"), "--limit", "3"])

    assert len(list(read_manifest(tmp_path / "m.jsonl"))) == 3
    assert "3 objetos" in capsys.readouterr().out


def test_audio_manifest_uses_the_same_segmentation_rule(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    # Una cola de 60 muestras (descartada) y un último segmento de 3 s (conservado).
    for name, n in [("tail.wav", 15 * SAMPLE_RATE + 60), ("ok.wav", 18 * SAMPLE_RATE)]:
        sf.write(root / name, np.zeros(n), SAMPLE_RATE)

    rows = list(audio_rows(root))

    for name in ("ok.wav", "tail.wav"):
        track_rows = [r for r in rows if r["path"] == name]
        segments = list(iter_audio_segments(root / name))
        assert len(track_rows) == len(segments) == count_audio_segments(root / name)
        assert [r["segment"] for r in track_rows] == [i for i, _ in segments]
    assert [(r["path"], r["end"]) for r in rows] == [
        ("ok.wav", 5.0),
        ("ok.wav", 10.0),
        ("ok.wav", 15.0),
        ("ok.wav", 18.0),
        ("tail.wav", 5.0),
        ("tail.wav", 10.0),
        ("tail.wav", 15.0),
    ]
