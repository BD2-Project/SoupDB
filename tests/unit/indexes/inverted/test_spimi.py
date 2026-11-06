from pathlib import Path

import pytest

from engine.algorithms.external_sort import write_sorted_run
from engine.common.rid import RID
from engine.indexes.inverted.merge import (
    compare_postings,
    decode_posting,
    encode_posting,
    merged_terms,
)
from engine.indexes.inverted.spimi import spimi_runs
from engine.indexes.inverted.tokenizer import Tokenizer

TOKENIZER = Tokenizer(stopwords=frozenset(), stemming=False)


def test_blocks_and_single_block_produce_same_frequencies(tmp_path: Path) -> None:
    items = [("Árbol disk disk", RID(8, 3)), ("disk árbol page", RID(2, 1)), ("", RID(9, 0))]
    small = spimi_runs(items, tmp_path, tokenizer=TOKENIZER, memory_limit_bytes=1)
    large = spimi_runs(items, tmp_path, tokenizer=TOKENIZER, memory_limit_bytes=100000)
    assert len(small.runs) == 5
    assert len(large.runs) == 1
    expected = [("disk", [(0, 2), (1, 1)]), ("page", [(1, 1)]), ("árbol", [(0, 1), (1, 1)])]
    assert list(merged_terms(small.runs, fan_in=2, temp_dir=tmp_path)) == expected
    assert list(merged_terms(large.runs)) == expected
    assert small.doc_lengths == large.doc_lengths == {0: 3, 1: 3, 2: 0}
    assert small.rids == {0: RID(8, 3), 1: RID(2, 1), 2: RID(9, 0)}


def test_merge_combines_document_frequency_split_across_runs(tmp_path: Path) -> None:
    a = write_sorted_run(tmp_path / "a", [encode_posting("term", 1, 2)], compare_postings)
    b = write_sorted_run(tmp_path / "b", [encode_posting("term", 1, 3)], compare_postings)
    assert list(merged_terms([a, b])) == [("term", [(1, 5)])]


def test_empty_and_stopword_documents_are_counted(tmp_path: Path) -> None:
    result = spimi_runs(
        [("", RID(0, 0)), ("the and", RID(0, 1))],
        tmp_path,
        tokenizer=Tokenizer(),
        memory_limit_bytes=100,
    )
    assert result.runs == ()
    assert result.doc_lengths == {0: 0, 1: 0}
    assert list(merged_terms(result.runs)) == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_budget_does_not_create_outputs(tmp_path: Path, invalid) -> None:
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        spimi_runs([], tmp_path, tokenizer=TOKENIZER, memory_limit_bytes=invalid)
    assert list(tmp_path.iterdir()) == []


def test_failure_cleans_only_owned_runs(tmp_path: Path) -> None:
    sentinel = tmp_path / "keep"
    sentinel.write_text("unchanged")
    with pytest.raises(ValueError, match="duplicate"):
        spimi_runs(
            [("first", RID(1, 0)), ("duplicate", RID(1, 0))],
            tmp_path,
            tokenizer=TOKENIZER,
            memory_limit_bytes=1,
        )
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"


def test_input_failure_cleans_written_runs(tmp_path: Path) -> None:
    def documents():
        yield "first", RID(1, 0)
        raise RuntimeError("input failed")

    with pytest.raises(RuntimeError, match="input failed"):
        spimi_runs(documents(), tmp_path, tokenizer=TOKENIZER, memory_limit_bytes=1)
    assert list(tmp_path.iterdir()) == []


def test_early_close_of_grouped_merge_cleans_temp(tmp_path: Path) -> None:
    result = spimi_runs(
        [("alpha beta gamma delta", RID(1, 0))],
        tmp_path,
        tokenizer=TOKENIZER,
        memory_limit_bytes=1,
    )
    work = tmp_path / "merge"
    work.mkdir()
    groups = merged_terms(result.runs, fan_in=2, temp_dir=work)
    assert next(groups) == ("alpha", [(0, 1)])
    groups.close()
    assert list(work.iterdir()) == []
    for path in result.runs:
        path.unlink()


def test_posting_codec_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        decode_posting(encode_posting("term", 0, 0))
