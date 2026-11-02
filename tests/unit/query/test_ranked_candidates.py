"""Candidate policy guarantees no silent truncation after filtering or sorting."""

from dataclasses import replace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.query.ranked_candidates import CandidateRequest, choose_candidates
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


@pytest.fixture
def binding():
    return RankedBinding(
        "idx_text",
        "documentos",
        "contenido",
        "INVERTED",
        ScriptedRankedIndex("text"),
        item_count=12,
        exact=True,
        exhaustive=True,
    )


def test_top_k_limit_and_offset(binding):
    assert choose_candidates(binding, limit=4, offset=3) == CandidateRequest(7, "bounded_top_k")


def test_limit_zero_does_not_require_provider_capabilities(binding):
    limited = replace(binding, exact=False, exhaustive=False, item_count=None)
    assert choose_candidates(limited, limit=0, filtered=True) == CandidateRequest(0, "empty")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": None},
        {"limit": 2, "filtered": True},
        {"limit": 2, "distinct": True},
        {"limit": 2, "reordered": True},
    ],
)
def test_full_candidate_cases_request_item_count(binding, kwargs):
    assert choose_candidates(binding, **kwargs) == CandidateRequest(12, "exhaustive")


@pytest.mark.parametrize(
    "changes",
    [
        {"exact": False},
        {"exhaustive": False},
        {"item_count": None},
    ],
)
def test_capability_gaps_rejected_for_exhaustive_queries(binding, changes):
    with pytest.raises(QueryExecutionError, match="exact exhaustive"):
        choose_candidates(replace(binding, **changes), limit=None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": -1},
        {"limit": 1.5},
        {"limit": True},
        {"limit": 2, "offset": -1},
        {"limit": 2, "offset": True},
    ],
)
def test_invalid_limit_rejected(binding, kwargs):
    with pytest.raises(QueryExecutionError):
        choose_candidates(binding, **kwargs)
