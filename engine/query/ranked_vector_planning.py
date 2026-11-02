"""Vector-ranked SQL planner, backed by injected profiles and RankedIndex."""

from typing import Any

from engine.common.errors import QueryExecutionError
from engine.query.ast import (
    BoundScoreExpr,
    Expr,
    FunctionExpr,
    SelectStatement,
    SimilarToExpr,
)
from engine.query.operators import Limit, Project, Sort
from engine.query.ranked_binding import visible_projections
from engine.query.ranked_planning import _order_keys, _validate_scalar, _walk_nodes
from engine.query.vector_operators import HNSWSearch, IVFSearch


def _get_similarity(predicate: Expr | None) -> SimilarToExpr:
    if not isinstance(predicate, SimilarToExpr):
        if predicate is not None and any(
            isinstance(node, SimilarToExpr) for node in _walk_nodes(predicate)
        ):
            raise QueryExecutionError("filtered/combined SIMILAR_TO is not supported")
        raise QueryExecutionError("vector search requires exactly one SIMILAR_TO predicate")
    return predicate


def plan_vector_search(statement: SelectStatement, catalog: Any):
    """Return a safe ANN top-k over the k *global* matches; no hidden filters."""
    options = statement.search
    if options is None or options.strategy not in ("IVF", "HNSW"):
        raise QueryExecutionError("SIMILAR_TO requires USING IVF or HNSW")
    if statement.group_by or statement.having is not None or statement.joins:
        raise QueryExecutionError("SIMILAR_TO does not support GROUP BY, HAVING or JOIN")
    if statement.distinct:
        raise QueryExecutionError("SIMILAR_TO with DISTINCT is not supported")
    if any(
        isinstance(node, FunctionExpr)
        for col in statement.columns
        for node in _walk_nodes(col.expr)
    ):
        raise QueryExecutionError("aggregates are not supported with SIMILAR_TO")
    sim = _get_similarity(statement.where)
    k = sim.k.value
    if type(k) is not int or k < 0:
        raise QueryExecutionError("SIMILAR_TO k must be a non-negative integer")
    schema = catalog.schema(statement.table)
    metric = options.metric
    matches = [
        binding
        for binding in catalog.ranked_indexes_for(statement.table).values()
        if binding.algorithm == options.strategy and binding.metric == metric
    ]
    if not matches:
        raise QueryExecutionError(
            f"no {options.strategy} index with metric {metric!r} for {statement.table!r}"
        )
    if len(matches) != 1:
        names = ", ".join(sorted(binding.name for binding in matches))
        raise QueryExecutionError(f"ambiguous vector indexes: {names}")
    binding = matches[0]
    vectorizer = getattr(catalog, "query_vectorizer", None)
    if not callable(getattr(vectorizer, "vectorize", None)):
        raise QueryExecutionError("query vectorizer is not configured")

    selections = visible_projections(statement.columns, schema, position=len(schema), kind="vector")
    for selection in selections:
        if not isinstance(selection.expr, BoundScoreExpr):
            _validate_scalar(selection.expr, schema)
    order_by, _ = _order_keys(statement, schema, kind="vector")

    cls = IVFSearch if options.strategy == "IVF" else HNSWSearch
    disk = getattr(catalog, "disk_manager", None)
    root = cls(
        binding,
        catalog.file_org(statement.table),
        sim.source.value,
        schema,
        request=k,
        vectorizer=vectorizer,
        disk_manager=disk,
    )
    if order_by:
        root = Sort(root, order_by, disk_manager=disk)
    root = Project(root, selections, disk_manager=disk)
    if statement.limit is not None:
        limit = statement.limit.limit.value
        offset = statement.limit.offset.value if statement.limit.offset else 0
        root = Limit(root, limit, offset, disk_manager=disk)
    return root
