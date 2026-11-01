"""Catalog-independent metadata for SQL ranked-search indexes.

This module complements ``RankedIndex``; it does not change the existing
scalar ``Index`` interface or implement index persistence.
"""

from dataclasses import dataclass

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.base import RankedIndex

_RANKED_ALGORITHMS = frozenset({"INVERTED", "IVF", "HNSW"})
_VECTOR_METRICS = frozenset({"euclidean", "cosine", "inner_product"})
_REFERENCE_TYPES = frozenset({ColumnType.TEXT, ColumnType.VARCHAR})


@dataclass(frozen=True, slots=True)
class RankedBinding:
    """Describe a ranked index attached to a table column.

    ``algorithm`` identifies the index family; text scoring methods such as
    TF-IDF and BM25 are supplied to ``top_k`` and are not index families.

    Capabilities are explicit provider declarations, not guarantees inferred
    from the presence of a method or a particular index implementation.
    """

    name: str
    table: str
    column: str
    algorithm: str
    index: RankedIndex
    metric: str | None = None
    dimension: int | None = None
    profile_id: str | None = None
    item_count: int | None = None
    exact: bool = False
    exhaustive: bool = False
    mutable: bool = False


def validate_ranked_binding(binding: RankedBinding, schema: tuple[ColumnDef, ...]) -> None:
    """Validate ranked-index metadata against the owning table schema.

    Does not inspect the index's underlying files or assert that a declared
    capability is actually implemented; provider integration tests do that.
    """
    for field in ("name", "table", "column"):
        value = getattr(binding, field)
        if not isinstance(value, str) or not value.strip():
            raise QueryExecutionError(f"ranked binding {field} must be a non-empty string")

    if not isinstance(binding.algorithm, str) or binding.algorithm not in _RANKED_ALGORITHMS:
        raise QueryExecutionError(f"unsupported ranked algorithm {binding.algorithm!r}")
    if not isinstance(binding.index, RankedIndex):
        raise QueryExecutionError("ranked binding index must implement RankedIndex")

    column = next((col for col in schema if col.name == binding.column), None)
    if column is None:
        raise QueryExecutionError(
            f"unknown column {binding.column!r} for ranked index {binding.name!r}"
        )
    if column.type_name not in _REFERENCE_TYPES:
        raise QueryExecutionError(f"ranked index column {binding.column!r} must be TEXT or VARCHAR")

    if binding.algorithm == "INVERTED":
        if any(
            value is not None for value in (binding.metric, binding.dimension, binding.profile_id)
        ):
            raise QueryExecutionError("INVERTED bindings cannot have vector metadata")
    else:
        if not isinstance(binding.metric, str) or binding.metric not in _VECTOR_METRICS:
            raise QueryExecutionError(f"{binding.algorithm} requires a supported vector metric")
        if type(binding.dimension) is not int or binding.dimension <= 0:
            raise QueryExecutionError("vector dimension must be a positive integer")
        if not isinstance(binding.profile_id, str) or not binding.profile_id.strip():
            raise QueryExecutionError("vector profile_id must be a non-empty string")

    if binding.item_count is not None and (
        type(binding.item_count) is not int or binding.item_count < 0
    ):
        raise QueryExecutionError("item_count must be a non-negative integer or None")
    for field in ("exact", "exhaustive", "mutable"):
        if type(getattr(binding, field)) is not bool:
            raise QueryExecutionError(f"{field} capability must be boolean")
