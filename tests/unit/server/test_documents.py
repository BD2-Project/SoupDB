"""Tests for secure document reference resolution (server layer)."""

import pytest

from engine.common.errors import RecordNotFound, UnsupportedOperation
from server.documents import DocumentStore, media_type_for


@pytest.fixture
def root(tmp_path):
    (tmp_path / "gatos").mkdir()
    (tmp_path / "gatos" / "001.jpg").write_bytes(b"jpeg-bytes")
    (tmp_path / "audio.mp3").write_bytes(b"mp3-bytes")
    return tmp_path


def test_resolves_relative_reference(root) -> None:
    document = DocumentStore(root).resolve("gatos/001.jpg")

    assert document.path == (root / "gatos" / "001.jpg").resolve()
    assert document.media_type == "image/jpeg"
    assert document.size == len(b"jpeg-bytes")
    assert document.filename == "001.jpg"


def test_reads_document_in_chunks(root) -> None:
    document = DocumentStore(root).resolve("gatos/001.jpg")

    assert b"".join(document.read_chunks(chunk_size=4)) == b"jpeg-bytes"


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "   ",
        "/etc/passwd",
        "C:/x.jpg",
        "gatos\\001.jpg",
        "../secret.jpg",
        "gatos/../gatos/001.jpg",
        "./gatos/001.jpg",
        "gatos//001.jpg",
    ],
)
def test_rejects_invalid_references(root, reference) -> None:
    with pytest.raises(RecordNotFound):
        DocumentStore(root).resolve(reference)


def test_rejects_missing_file(root) -> None:
    with pytest.raises(RecordNotFound):
        DocumentStore(root).resolve("nope.jpg")


def test_rejects_directory(root) -> None:
    with pytest.raises(RecordNotFound):
        DocumentStore(root).resolve("gatos")


def test_rejects_root_that_is_not_a_directory(tmp_path) -> None:
    with pytest.raises(UnsupportedOperation):
        DocumentStore(tmp_path / "missing")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.JPG", "image/jpeg"),
        ("a.jpeg", "image/jpeg"),
        ("a.png", "image/png"),
        ("a.mp3", "audio/mpeg"),
        ("a.wav", "audio/wav"),
        ("a.mp4", "video/mp4"),
        ("a.txt", "text/plain"),
        ("a.unknown", "application/octet-stream"),
    ],
)
def test_media_type_for(name, expected) -> None:
    assert media_type_for(name) == expected
