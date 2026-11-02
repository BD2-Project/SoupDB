"""Bind and plan ranked full-text SELECT using an injected catalog.

The fake and eventual persistent catalog use the same ranked_indexes_for
contract. No full-text scoring, tokenizer, or table scan is implemented here.
"""

from dataclasses import fields, is_dataclass
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnType
from engine.query.ast import (
    BoundScoreExpr,
    ColumnRef,
    Expr,
    FunctionExpr,
    Literal,
    LogicalExpr,
    MatchExpr,
    OrderByItem,
    ScoreExpr,
    SelectStatement,
)
from engine.query.operators import Distinct, Filter, Limit, Project, Sort
from engine.query.ranked_binding import bind_score, visible_projections
from engine.query.ranked_candidates import choose_candidates
from engine.query.ranked_operators import InvertedIndexScan


def _walk_nodes(value: object):
    if isinstance(value, Expr):
        yield value
    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_nodes(child)
    elif is_dataclass(value):
        for field in fields(value):
            yield from _walk_nodes(getattr(value, field.name))


def _split_match(predicate: Expr | None) -> tuple[MatchExpr, Expr | None]:
    """Split conjunction into its single MATCH and a residual predicate."""
    predicates: list[Expr] = []

    def gather(node: Expr) -> None:
        if isinstance(node, LogicalExpr) and node.op == "AND":
            gather(node.left)
            gather(node.right)
        else:
            predicates.append(node)

    if predicate is None:
        raise QueryExecutionError("ranked search requires a MATCH predicate")
    gather(predicate)
    matches = [value for value in predicates if isinstance(value, MatchExpr)]
    if len(matches) != 1:
        raise QueryExecutionError("ranked search requires exactly one MATCH predicate")
    remaining = [node for node in predicates if not isinstance(node, MatchExpr)]
    if any(isinstance(expr, MatchExpr) for node in remaining for expr in _walk_nodes(node)):
        raise QueryExecutionError("MATCH inside OR/NOT is not supported")
    residual: Expr | None = None
    for node in remaining:
        residual = node if residual is None else LogicalExpr(residual, "AND", node)
    return matches[0], residual


def _validate_scalar(expr: Expr, schema) -> None:
    names = {col.name for col in schema}
    for node in _walk_nodes(expr):
        if isinstance(node, ColumnRef) and node.name not in names:
            raise QueryExecutionError(f"unknown column {node.name!r}")
        if isinstance(node, (ScoreExpr, BoundScoreExpr, MatchExpr, FunctionExpr)):
            raise QueryExecutionError(
                "SCORE/MATCH/aggregates cannot be nested in a residual predicate"
            )


def _order_keys(statement: SelectStatement, schema) -> tuple[tuple[OrderByItem, ...], bool]:
    """Bind SCORE aliases and flag ordering that requires all candidates."""
    aliases = {col.alias: col.expr for col in statement.columns if col.alias}
    position = len(schema)
    result: list[OrderByItem] = []
    reordered = False
    for item in statement.order_by:
        expr = item.expr
        if isinstance(expr, ColumnRef) and expr.name in aliases:
            expr = aliases[expr.name]
        if isinstance(expr, ScoreExpr):
            expr = bind_score(expr, position=position, kind="text")
        elif any(
            isinstance(node, (ScoreExpr, BoundScoreExpr, MatchExpr)) for node in _walk_nodes(expr)
        ):
            raise QueryExecutionError("nested SCORE/MATCH ordering is not supported")
        else:
            _validate_scalar(expr, schema)
        if not isinstance(expr, BoundScoreExpr) or item.ascending:
            reordered = True
        result.append(OrderByItem(expr, item.ascending))
    return tuple(result), reordered


def plan_text_search(statement: SelectStatement, catalog: Any) -> Any:
    """Build a correct ranked plan or fail explicitly on missing capabilities."""
    if statement.search is None or statement.search.strategy not in ("BM25", "TF_IDF"):
        raise QueryExecutionError("MATCH requires USING BM25 or USING TF_IDF")
    if statement.group_by or statement.having is not None or statement.joins:
        raise QueryExecutionError("MATCH does not yet support GROUP BY, HAVING or JOIN")
    if any(
        isinstance(node, FunctionExpr)
        for col in statement.columns
        for node in _walk_nodes(col.expr)
    ):
        raise QueryExecutionError("aggregates are not yet supported with MATCH")

    match, residual = _split_match(statement.where)
    schema = catalog.schema(statement.table)
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

    selections = visible_projections(statement.columns, schema, position=len(schema), kind="text")
    for selection in selections:
        if not isinstance(selection.expr, BoundScoreExpr):
            _validate_scalar(selection.expr, schema)
    if residual is not None:
        _validate_scalar(residual, schema)
    order_by, reordered = _order_keys(statement, schema)

    limit = statement.limit.limit.value if statement.limit else None
    offset = statement.limit.offset.value if statement.limit and statement.limit.offset else 0
    policy = choose_candidates(
        binding,
        limit=limit,
        offset=offset,
        filtered=residual is not None,
        distinct=statement.distinct,
        reordered=reordered,
    )
    root = InvertedIndexScan(
        binding,
        catalog.file_org(statement.table),
        match.query.value,
        schema,
        request=policy.count,
        method=statement.search.strategy.lower(),
        disk_manager=disk,
        candidate_policy=policy.policy,
    )
    if residual is not None:
        root = Filter(root, residual, root.schema, disk)
    if order_by:
        root = Sort(root, order_by, disk_manager=disk)
    root = Project(root, selections, disk_manager=disk)
    if statement.distinct:
        root = Distinct(root, None, disk)
    if limit is not None:
        root = Limit(root, limit, offset, disk_manager=disk)
    return root
