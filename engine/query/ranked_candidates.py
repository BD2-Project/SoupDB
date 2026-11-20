"""Candidate cardinality policy for text-ranked SQL."""

from dataclasses import dataclass

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding


@dataclass(frozen=True)
class CandidateRequest:
    count: int
    policy: str


def choose_candidates(
    binding: RankedBinding,
    *,
    limit: int | None,
    offset: int = 0,
    filtered: bool = False,
    distinct: bool = False,
    reordered: bool = False,
) -> CandidateRequest:
    """Choose exact requested k, refusing unsafe post-top-k semantics."""
    if limit is not None and (type(limit) is not int or limit < 0):
        raise QueryExecutionError("ranked LIMIT must be a non-negative integer")
    if type(offset) is not int or offset < 0:
        raise QueryExecutionError("ranked OFFSET must be a non-negative integer")

    # Zero output is always correct; do not open the provider unnecessarily.
    if limit == 0:
        return CandidateRequest(0, "empty")

    if limit is not None and not (filtered or distinct or reordered):
        return CandidateRequest(limit + offset, "bounded_top_k")

    if not binding.exact or not binding.exhaustive or binding.item_count is None:
        raise QueryExecutionError(
            "ranked search requires exact exhaustive candidates and a known "
            "item_count for no LIMIT, residual filters, DISTINCT or reordering"
        )
    return CandidateRequest(binding.item_count, "exhaustive")
