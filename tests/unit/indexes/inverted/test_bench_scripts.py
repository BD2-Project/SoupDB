import hashlib
import json
from pathlib import Path

import pytest

from scripts import bench_inverted
from scripts.sample_inverted_corpus import reservoir_sample, write_sample


def _lines(count: int) -> list[str]:
    return [json.dumps({"_id": str(i), "title": "Índice", "text": f"database query {i}"})
            for i in range(count)]


def test_reservoir_reproducible_seeded_and_preserves_schema() -> None:
    sample, count = reservoir_sample(_lines(100), size=10, seed=7)
    assert count == 100
    assert len({row["_id"] for row in sample}) == 10
    assert sample == reservoir_sample(_lines(100), size=10, seed=7)[0]
    assert sample != reservoir_sample(_lines(100), size=10, seed=8)[0]
    assert all(set(row) == {"_id", "title", "text"} for row in sample)
    assert any(int(row["_id"]) >= 10 for row in sample)


@pytest.mark.parametrize("line", ['{"_id":"x","text":"missing title"}',
                                 '{"_id":true,"title":"x","text":"x"}',
                                 '{"_id":"x","title":3,"text":"x"}', '[]', '{bad'])
def test_reservoir_rejects_malformed_schema(line: str) -> None:
    with pytest.raises(ValueError, match="line 1"):
        reservoir_sample([line], size=1)


def test_short_source_and_invalid_size_fail() -> None:
    with pytest.raises(ValueError, match="requested 2"):
        reservoir_sample(_lines(1), size=2)
    with pytest.raises(ValueError):
        reservoir_sample([], size=0)


def test_export_is_reproducible_utf8_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    output = tmp_path / "sample.jsonl"
    sample, _ = reservoir_sample(_lines(5), size=3)
    digest = write_sample(output, sample)
    first = output.read_bytes()
    assert digest == hashlib.sha256(first).hexdigest()
    assert write_sample(output, sample) == digest
    assert output.read_bytes() == first
    assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == sample
    assert list(tmp_path.iterdir()) == [output]


def test_benchmark_reopens_source_ids_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "sample.jsonl"
    write_sample(corpus, [
        {"_id": "paper-a", "title": "database", "text": "index index"},
        {"_id": "paper-b", "title": "ocean", "text": "water"},
    ])
    created = []
    original = bench_inverted.tempfile.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        result = original(*args, **kwargs)
        created.append(Path(result.name))
        return result

    monkeypatch.setattr(bench_inverted.tempfile, "TemporaryDirectory", temporary_directory)
    report = bench_inverted.benchmark(corpus, queries=("database", "unknownterm"), k=10)
    assert report["documents"] == 2
    assert report["reopen_verified"] is True
    assert report["artifact_bytes"] > 0
    assert all(not path.exists() for path in created)
    for result in report["queries"]:
        if result["query"] == "database":
            assert [hit["source_id"] for hit in result["hits"]] == ["paper-a"]
        else:
            assert result["hits"] == []
    with pytest.raises(ValueError):
        bench_inverted.benchmark(corpus, memory_limit_bytes=0)
    assert all(not path.exists() for path in created)
