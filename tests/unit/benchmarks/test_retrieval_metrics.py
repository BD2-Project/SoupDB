import json
from pathlib import Path

import pytest

from benchmarks.retrieval_metrics import (
    build_query_length_ablation,
    parse_qrels,
    precision_at_k,
    recall_at_k,
    restrict_qrels_to_corpus,
)
from engine.indexes.inverted.tokenizer import DEFAULT_TOKENIZER


def _qrel(qid, docid, grade):
    return json.dumps({"query-id": str(qid), "corpus-id": str(docid), "score": grade})


def test_parse_positive_graded_judgments_and_ignore_zero_unjudged():
    qrels = parse_qrels([
        _qrel("1", "partial", 1), _qrel("1", "full", 2),
        _qrel("1", "nonrelevant", 0), _qrel("1", "unjudged", -1),
    ])
    assert qrels == {"1": frozenset({"partial", "full"})}


def test_accept_bytes_blank_lines_and_repeated_consistent_qrels():
    qrels = parse_qrels([_qrel("01", "0007", 1).encode(), b"\n", _qrel("01", "0007", 1)])
    assert qrels == {"01": frozenset({"0007"})}


def test_qrels_accept_numeric_grades_encoded_as_strings():
    rows = [
        json.dumps({"query-id": "q", "corpus-id": "d1", "score": "2"}),
        json.dumps({"query-id": "q", "corpus-id": "d2", "score": "0.0"}),
    ]
    assert parse_qrels(rows) == {"q": frozenset({"d1"})}


@pytest.mark.parametrize("rows", [
    ["{"], ["[]"], [_qrel("", "d", 1)], [_qrel("q", "", 1)],
    [_qrel("q", "d", -2)], [_qrel("q", "d", 3)],
    [json.dumps({"query-id": "q", "corpus-id": "d", "score": True})],
    [_qrel("q", "d", 1), _qrel("q", "d", 2)],
])
def test_invalid_or_conflicting_qrels_are_rejected(rows):
    with pytest.raises(ValueError):
        parse_qrels(rows)


def test_metrics_are_conditional_on_judged_relevants_inside_sample():
    qrels = {"q1": {"a", "outside"}, "q2": {"outside"}}
    eligible = restrict_qrels_to_corpus(qrels, {"a", "b", "c"})
    assert eligible == {"q1": frozenset({"a"})}
    assert precision_at_k(["a", "b"], eligible["q1"], k=10) == pytest.approx(0.1)
    assert recall_at_k(["a", "b"], eligible["q1"], k=10) == 1.0
    assert recall_at_k([], (), k=10) is None


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_metric_k_must_be_positive_integer(k):
    with pytest.raises(ValueError):
        precision_at_k([], (), k)
    with pytest.raises(ValueError):
        recall_at_k([], (), k)


def test_trec_covid_query_ablation_preserves_48_topic_ids_and_exact_buckets():
    query_file = Path("benchmarks/data/trec_covid_queries.jsonl")
    rows = query_file.read_text(encoding="utf-8").splitlines()
    variants = build_query_length_ablation(rows)
    expected_ids = {
        row["_id"] for row in map(json.loads, rows)
        if len(DEFAULT_TOKENIZER(row["text"])) >= 5
    }
    assert len(expected_ids) == 48
    assert set(variants) == {"1", "3", "5+"}
    assert all(set(bucket) == expected_ids for bucket in variants.values())
    assert all(len(DEFAULT_TOKENIZER(query)) == 1 for query in variants["1"].values())
    assert all(len(DEFAULT_TOKENIZER(query)) == 3 for query in variants["3"].values())
    assert all(len(DEFAULT_TOKENIZER(query)) >= 5 for query in variants["5+"].values())


def test_query_ablation_accepts_decoded_rows_and_rejects_duplicate_ids():
    rows = [{"_id": "x", "text": "Longer topic about testing queries"}]
    result = build_query_length_ablation(rows)
    assert set(result["1"]) == {"x"}
    assert result["3"]["x"] == "longer topic about"
    with pytest.raises(ValueError, match="duplicate"):
        build_query_length_ablation(rows * 2)
