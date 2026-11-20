"""TREC-COVID qrels parsing and binary Precision@k / conditional Recall@k."""

import json
import math
from collections.abc import Iterable, Mapping, Sequence

from engine.indexes.inverted.tokenizer import DEFAULT_TOKENIZER, Tokenizer

Qrels = dict[str, frozenset[str]]


def build_query_length_ablation(
    rows: Iterable[str | bytes | Mapping[str, object]],
    *,
    tokenizer: Tokenizer = DEFAULT_TOKENIZER,
) -> dict[str, dict[str, str]]:
    """Create controlled 1/3/5+ token variants for MTEB query rows.

    Only original topics with at least five tokens are eligible. ``1`` and
    ``3`` are prefixes of the tokenized original; ``5+`` keeps the original
    text. Each bucket maps the original query ID to its derived query text, so
    all variants reuse the parent topic's qrels. The shortened variants are
    query reformulations, not independently judged queries.

    Rows may be MTEB JSONL lines or already decoded mappings with ``_id`` and
    ``text`` fields. Output bucket names are ``"1"``, ``"3"``, and ``"5+"``.
    """
    variants: dict[str, dict[str, str]] = {"1": {}, "3": {}, "5+": {}}
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        if isinstance(row, (str, bytes)):
            if isinstance(row, bytes):
                row = row.decode("utf-8")
            if not row.strip():
                continue
            try:
                row = json.loads(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid query JSON at line {line_number}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"query row {line_number} must be an object")
        query_id, text = row.get("_id"), row.get("text")
        if not isinstance(query_id, str) or not query_id.strip() or not isinstance(text, str):
            raise ValueError(f"invalid query fields at line {line_number}")
        if query_id in seen:
            raise ValueError(f"duplicate query ID at line {line_number}: {query_id}")
        seen.add(query_id)
        tokens = tokenizer(text)
        if len(tokens) < 5:
            continue
        variants["1"][query_id] = tokens[0]
        variants["3"][query_id] = " ".join(tokens[:3])
        variants["5+"][query_id] = text
    return variants


def parse_qrels(lines: Iterable[str | bytes]) -> Qrels:
    """Parse MTEB JSONL qrels, keeping grades 1/2 as relevant.

    Grade 0 is judged nonrelevant and -1 means unjudged; both are excluded
    from the relevant set. IDs stay strings so they match corpus `_id` exactly.
    Repeated (query, document) pairs are accepted only if grades agree.
    """
    grades: dict[tuple[str, str], int] = {}
    for line_number, line in enumerate(lines, 1):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid qrels JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"qrels row {line_number} must be an object")
        query_id, document_id, grade = (row.get("query-id"), row.get("corpus-id"), row.get("score"))
        if (
            not isinstance(query_id, str)
            or not query_id.strip()
            or not isinstance(document_id, str)
            or not document_id.strip()
            or type(grade) not in (int, float, str)
        ):
            raise ValueError(f"invalid qrels fields at line {line_number}")
        try:
            numeric_grade = float(grade)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid qrels fields at line {line_number}") from exc
        if not math.isfinite(numeric_grade) or numeric_grade not in (-1, 0, 1, 2):
            raise ValueError(f"invalid qrels fields at line {line_number}")
        key = (query_id, document_id)
        integer_grade = int(numeric_grade)
        if key in grades and grades[key] != integer_grade:
            raise ValueError(f"conflicting qrels judgments at line {line_number}")
        grades[key] = integer_grade

    relevant: dict[str, set[str]] = {}
    for (query_id, document_id), grade in grades.items():
        if grade > 0:
            relevant.setdefault(query_id, set()).add(document_id)
    return {query_id: frozenset(document_ids) for query_id, document_ids in relevant.items()}


def restrict_qrels_to_corpus(
    qrels: Mapping[str, Iterable[str]], corpus_ids: Iterable[str]
) -> Qrels:
    """Use only judged relevant documents present in a sampled corpus.

    Recall from this result is conditional on the sample containing a judged
    relevant document. Queries with no eligible relevant documents are omitted.
    """
    available = set(corpus_ids)
    return {
        query_id: eligible
        for query_id, documents in qrels.items()
        if (eligible := frozenset(documents).intersection(available))
    }


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Binary P@k with fixed denominator k, including missing result slots."""
    _validate_k(k)
    relevant_ids = set(relevant)
    return sum(document_id in relevant_ids for document_id in retrieved[:k]) / k


def recall_at_k(
    retrieved: Sequence[str], relevant_in_corpus: Iterable[str], k: int = 10
) -> float | None:
    """Recall@k against judged relevant documents inside this corpus sample.

    Returns None when the query has no judged relevant documents in the sample;
    callers should omit that query from the macro average and report its count.
    """
    _validate_k(k)
    eligible = set(relevant_in_corpus)
    if not eligible:
        return None
    return len(set(retrieved[:k]).intersection(eligible)) / len(eligible)


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
