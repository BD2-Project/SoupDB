"""Volcano leaf operators for ranked index hits backed by real table RIDs.

The ranked index is shared/borrowed: neither closing this operator nor a failed
query closes the binding's RankedIndex instance.
"""

from collections.abc import Mapping
from math import isfinite
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding, validate_ranked_binding
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.indexes.base import Query
from engine.query.evaluator import Schema
from engine.query.operators import PlanNode, _VolcanoBase
from engine.query.ranked_binding import append_score_schema
from engine.storage.base import FileOrganization
from engine.storage.disk_manager import DiskManager


def _validate_hit(hit: object) -> tuple[RID, float]:
    """Validate one provider (RID, score) pair without altering its score."""
    if not isinstance(hit, tuple) or len(hit) != 2:
        raise QueryExecutionError("ranked index returned an invalid (RID, score) hit")
    rid, score = hit
    if (
        not isinstance(rid, RID)
        or type(rid.page_id) is not int
        or type(rid.slot) is not int
        or rid.page_id < 0
        or rid.slot < 0
    ):
        raise QueryExecutionError("ranked index returned an invalid RID")
    if type(score) is not float or not isfinite(score):
        raise QueryExecutionError("ranked index score must be a finite FLOAT")
    return rid, score


class RankedScan(_VolcanoBase):
    """Read pre-ranked hits and materialize temporary rows with a FLOAT score.

    ``request`` is the exact number of candidates to request from top_k.
    Cardinality/exhaustiveness decisions are made by the planner, not here.
    """

    def __init__(
        self,
        binding: RankedBinding,
        file_org: FileOrganization,
        query: Query,
        schema: Schema,
        request: int,
        disk_manager: DiskManager | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        op_name: str = "RankedScan",
        candidate_policy: str = "bounded_top_k",
    ) -> None:
        super().__init__((), disk_manager)
        validate_ranked_binding(binding, schema)
        if type(request) is not int or request < 0:
            raise QueryExecutionError("ranked request must be a non-negative integer")
        self._binding = binding
        self._file_org = file_org
        self._query = query
        self._request = request
        self._options = dict(options or {})
        self._op_name = op_name
        self._candidate_policy = candidate_policy
        self._hits: tuple[tuple[RID, float], ...] = ()
        self._cursor = 0
        self._opened = False
        self.schema = append_score_schema(schema)
        self._base_schema = schema

    def open(self) -> None:
        """Acquire the shared table lock before calling the ranked provider."""
        if self._opened:
            raise QueryExecutionError("ranked scan is already open")
        super().open()
        self._hits = ()
        self._cursor = 0
        try:
            self._file_org.lock_shared()
            if self._request:
                raw_hits = self._search()
                if not isinstance(raw_hits, list):
                    raise QueryExecutionError("ranked index must return a list of hits")
                if len(raw_hits) > self._request:
                    raise QueryExecutionError("ranked index returned more hits than requested")
                seen: set[RID] = set()
                validated: list[tuple[RID, float]] = []
                last_score: float | None = None
                for raw_hit in raw_hits:
                    rid, score = _validate_hit(raw_hit)
                    if rid in seen:
                        raise QueryExecutionError(f"ranked index returned duplicate RID {rid!r}")
                    if last_score is not None and score > last_score:
                        raise QueryExecutionError("ranked index returned unsorted scores")
                    seen.add(rid)
                    validated.append((rid, score))
                    last_score = score
                self._hits = tuple(validated)
        except Exception as exc:
            self.close()
            if isinstance(exc, QueryExecutionError):
                raise
            raise QueryExecutionError(
                f"ranked search failed for index {self._binding.name!r}: {exc}"
            ) from exc
        self._opened = True

    def _search(self) -> list[tuple[RID, float]]:
        return self._binding.index.top_k(self._query, self._request, **self._options)

    def next(self) -> Record | None:
        if not self._opened:
            raise QueryExecutionError("ranked scan is not open")
        if self._cursor >= len(self._hits):
            return None
        rid, score = self._hits[self._cursor]
        self._cursor += 1
        try:
            record = self._file_org.fetch(rid)
            if record is None:
                raise QueryExecutionError(
                    f"stale RID {rid!r} in ranked index {self._binding.name!r}"
                )
            if not isinstance(record, Record):
                raise QueryExecutionError("ranked file organization returned a non-Record")
            row = decode_row(record.data, self._base_schema)
            result = Record(data=encode_row((*row, score), self.schema))
        except Exception as exc:
            if isinstance(exc, QueryExecutionError):
                raise
            raise QueryExecutionError(
                f"ranked record fetch failed for index {self._binding.name!r}: {exc}"
            ) from exc
        self._rows += 1
        return result

    def close(self) -> None:
        """Discard query-local state without closing the shared ranked index."""
        self._hits = ()
        self._cursor = 0
        self._opened = False
        super().close()

    def explain(self) -> PlanNode:
        return self._plan(
            self._op_name,
            {
                "index": self._binding.name,
                "table": self._binding.table,
                "column": self._binding.column,
                "algorithm": self._binding.algorithm,
                "requested_k": self._request,
                "candidate_policy": self._candidate_policy,
                "provider_hits": len(self._hits),
                "rows_fetched": self._rows,
                "exact": self._binding.exact,
                "exhaustive": self._binding.exhaustive,
                "options": dict(self._options),
            },
        )


class InvertedIndexScan(RankedScan):
    """Text ranked-search leaf for inverted indexes (TF-IDF or BM25)."""

    def _search(self) -> list[tuple[RID, float]]:
        # Real text providers distinguish MATCH candidates from the complete
        # RankedIndex universe. Existing providers/fakes retain their contract.
        match_search = getattr(self._binding.index, "match_top_k", None)
        if callable(match_search):
            return match_search(self._query, self._request, **self._options)
        return super()._search()

    def __init__(
        self,
        binding: RankedBinding,
        file_org: FileOrganization,
        query: str,
        schema: Schema,
        request: int,
        method: str,
        disk_manager: DiskManager | None = None,
        *,
        candidate_policy: str = "bounded_top_k",
    ) -> None:
        if binding.algorithm != "INVERTED":
            raise QueryExecutionError("InvertedIndexScan requires an INVERTED binding")
        if type(query) is not str:
            raise QueryExecutionError("text ranked search requires a string query")
        if method not in ("tf_idf", "bm25"):
            raise QueryExecutionError("text ranked method must be tf_idf or bm25")
        super().__init__(
            binding,
            file_org,
            query,
            schema,
            request,
            disk_manager,
            options={"method": method},
            op_name="InvertedIndexScan",
            candidate_policy=candidate_policy,
        )
