"""Conservative ranked full-text SELECT planning, independent of concrete indexes.

C07/C08: exact simple top-k, SCORE projections, and ORDER BY score.
More demanding candidate policies are added separately in C09.
"""

from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnType
from engine.query.ast import (
    BoundScoreExpr,
    ColumnRef,
    FunctionExpr,
    Literal,
    MatchExpr,
    OrderByItem,
    ScoreExpr,
    SelectStatement,
)
from engine.query.operators import Limit, Project, Sort
from engine.query.ranked_binding import visible_projections
from engine.query.ranked_operators import InvertedIndexScan


def plan_text_search(statement: SelectStatement, catalog: Any) -> Any:
    """Return a Volcano root for one constrained MATCH predicate."""
    if not isinstance(statement.where, MatchExpr):
        raise QueryExecutionError("ranked SQL currently needs a standalone MATCH predicate")
    if statement.search is None or statement.search.strategy not in ("BM25", "TF_IDF"):
        raise QueryExecutionError("MATCH requires USING BM25 or USING TF_IDF")
    if (
        statement.limit is None
        or statement.group_by
        or statement.having is not None
        or statement.joins
        or statement.distinct
    ):
        raise QueryExecutionError(
            "ranked SQL currently requires LIMIT without JOIN, GROUP BY or DISTINCT"
        )
    if any(isinstance(column.expr, FunctionExpr) for column in statement.columns):
        raise QueryExecutionError("aggregates are not yet supported with MATCH")

    schema = catalog.schema(statement.table)
    match = statement.where
    if not isinstance(match.query, Literal) or type(match.query.value) is not str:
        raise QueryExecutionError("MATCH requires a text query literal")
    column = next((col for col in schema if col.name == match.column.name), None)
    if column is None or column.type_name not in (ColumnType.TEXT, ColumnType.VARCHAR):
        raise QueryExecutionError("MATCH requires an existing TEXT or VARCHAR column")
    candidates = {
        name: binding
        for name, binding in catalog.ranked_indexes_for(statement.table, match.column.name).items()
        if binding.algorithm == "INVERTED"
    }
    if not candidates:
        raise QueryExecutionError(f"no INVERTED index for column {match.column.name!r}")
    if len(candidates) != 1:
        raise QueryExecutionError(f"ambiguous INVERTED indexes for column {match.column.name!r}")
    binding = next(iter(candidates.values()))
    disk = getattr(catalog, "disk_manager", None)
    position = len(schema)
    selections = visible_projections(statement.columns, schema, position=position, kind="text")

    # Limit and offset were checked syntactically by the parser; no default k.
    limit = statement.limit.limit.value
    offset = statement.limit.offset.value if statement.limit.offset else 0
    if type(limit) is not int or type(offset) is not int or limit < 0 or offset < 0:
        raise QueryExecutionError("ranked LIMIT/OFFSET must be non-negative integers")

    resolved_order: list[OrderByItem] = []
    aliases = {col.alias: col.expr for col in statement.columns if col.alias}
    for item in statement.order_by:
        expr = item.expr
        if isinstance(expr, ColumnRef) and expr.name in aliases:
            expr = aliases[expr.name]
        if isinstance(expr, ScoreExpr) and expr.kind == "text":
            expr = BoundScoreExpr(position, "text")
        if not isinstance(expr, BoundScoreExpr) or item.ascending is not False:
            raise QueryExecutionError("top-k MATCH only supports ORDER BY SCORE() DESC")
        resolved_order.append(OrderByItem(expr, ascending=False))

    root = InvertedIndexScan(
        binding,
        catalog.file_org(statement.table),
        match.query.value,
        schema,
        request=limit + offset,
        method=statement.search.strategy.lower(),
        disk_manager=disk,
    )
    if resolved_order:
        root = Sort(root, tuple(resolved_order), disk_manager=disk)
    root = Project(root, selections, disk_manager=disk)
    return Limit(root, limit, offset, disk_manager=disk)
