import hashlib
import json

import pytest

from scripts.sample_nested_corpora import nested_sample, write_nested_corpora


def _lines(count: int):
    for number in range(count):
        yield json.dumps({
            "_id": f"doc-{number:04d}",
            "title": f"Título {number}",
            "text": f"document text {number}",
            "ignored": "not copied",
        }, ensure_ascii=False)


def test_samples_are_reproducible_nested_and_preserve_only_schema_fields():
    first, count = nested_sample(_lines(120), sizes=(10, 30, 100), seed=7)
    second, second_count = nested_sample(_lines(120), sizes=(10, 30, 100), seed=7)
    other, _ = nested_sample(_lines(120), sizes=(10, 30, 100), seed=8)
    ids = [{row["_id"] for row in first[:size]} for size in (10, 30, 100)]

    assert count == second_count == 120
    assert first == second
    assert first[:100] != other[:100]
    assert ids[0] <= ids[1] <= ids[2]
    assert all(set(row) == {"_id", "title", "text"} for row in first)
    assert first[0]["title"].startswith("Título")


def test_writer_creates_hashed_manifest_and_nested_jsonl_files(tmp_path):
    output = tmp_path / "nested"
    manifest = write_nested_corpora(
        _lines(30), output, sizes=(5, 10, 25), seed=3, source="test-source"
    )
    assert manifest["source_documents"] == 30
    assert manifest["nested"] is True
    written_ids = []
    for size in (5, 10, 25):
        entry = manifest["samples"][str(size)]
        path = output / entry["file"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        assert len(rows) == size
        written_ids.append([row["_id"] for row in rows])
    assert set(written_ids[0]) <= set(written_ids[1]) <= set(written_ids[2])
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert {path.name for path in output.iterdir()} == {
        "manifest.json", "corpus_5.jsonl", "corpus_10.jsonl", "corpus_25.jsonl"
    }
    with pytest.raises(FileExistsError):
        write_nested_corpora(_lines(30), output, sizes=(5, 10, 25))


@pytest.mark.parametrize("sizes", [(), (10, 5), (5, 5), (0, 5), (True, 5)])
def test_invalid_size_configuration_is_rejected(sizes):
    with pytest.raises(ValueError):
        nested_sample(_lines(20), sizes=sizes)


def test_short_source_duplicate_ids_and_invalid_rows_fail(tmp_path):
    with pytest.raises(ValueError, match="requested"):
        nested_sample(_lines(2), sizes=(3,))
    duplicated = [
        json.dumps({"_id": "same", "title": "A", "text": "x"}),
        json.dumps({"_id": "same", "title": "B", "text": "y"}),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        nested_sample(duplicated, sizes=(1,))
    with pytest.raises(ValueError, match="line 1"):
        nested_sample(['{"_id":"x","title":null,"text":"bad"}'], sizes=(1,))

    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="duplicate"):
        write_nested_corpora(duplicated, target, sizes=(1,))
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
