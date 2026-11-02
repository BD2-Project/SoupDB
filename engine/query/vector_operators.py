"""IVF and HNSW SQL leaves, resolving references only at execution time."""

from math import isfinite
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.query.evaluator import Schema
from engine.query.ranked_operators import RankedScan
from engine.storage.base import FileOrganization
from engine.storage.disk_manager import DiskManager


class _VectorSearch(RankedScan):
    """RankedScan with an execution-time query-reference resolver.

    The vectorizer is injected. Parser/planner/EXPLAIN never open user files.
    """

    def __init__(
        self,
        binding: RankedBinding,
        file_org: FileOrganization,
        reference: str,
        schema: Schema,
        request: int,
        vectorizer: Any,
        disk_manager: DiskManager | None = None,
        *,
        candidate_policy: str = "bounded_top_k",
    ) -> None:
        if binding.algorithm not in ("IVF", "HNSW"):
            raise QueryExecutionError("vector search requires IVF or HNSW")
        if type(reference) is not str or not reference.strip():
            raise QueryExecutionError("SIMILAR_TO requires a non-empty reference")
        if not callable(getattr(vectorizer, "vectorize", None)):
            raise QueryExecutionError("query vectorizer is not configured")
        self._reference = reference
        self._vectorizer = vectorizer
        super().__init__(
            binding,
            file_org,
            (),
            schema,
            request,
            disk_manager,
            options={"metric": binding.metric},
            op_name=f"{binding.algorithm}Search",
            candidate_policy=candidate_policy,
        )

    def open(self) -> None:
        if self._opened:
            raise QueryExecutionError("ranked scan is already open")
        if self._request:
            try:
                values = self._vectorizer.vectorize(
                    self._reference, profile_id=self._binding.profile_id
                )
                if isinstance(values, (str, bytes)):
                    raise ValueError("vectorizer returned text instead of vector")
                vector = tuple(values)
                if len(vector) != self._binding.dimension:
                    raise ValueError(
                        f"vector dimension mismatch: expected {self._binding.dimension}, "
                        f"got {len(vector)}"
                    )
                if not all(type(x) in (int, float) and isfinite(x) for x in vector):
                    raise ValueError("query vector must contain finite numbers, not bool")
                self._query = tuple(float(x) for x in vector)
            except Exception as exc:
                raise QueryExecutionError(
                    f"failed to vectorize query for profile {self._binding.profile_id!r}: {exc}"
                ) from exc
        super().open()

    def explain(self):
        node = super().explain()
        node.detail.update(
            {
                "metric": self._binding.metric,
                "profile_id": self._binding.profile_id,
                "dimension": self._binding.dimension,
            }
        )
        return node


class IVFSearch(_VectorSearch):
    """IVF ranked SQL access path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        binding = args[0] if args else kwargs.get("binding")
        if binding is None or binding.algorithm != "IVF":
            raise QueryExecutionError("IVFSearch requires IVF binding")
        super().__init__(*args, **kwargs)


class HNSWSearch(_VectorSearch):
    """HNSW ranked SQL access path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        binding = args[0] if args else kwargs.get("binding")
        if binding is None or binding.algorithm != "HNSW":
            raise QueryExecutionError("HNSWSearch requires HNSW binding")
        super().__init__(*args, **kwargs)
