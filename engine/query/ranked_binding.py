"""Bind ranked-search scores to a private FLOAT slot in a temporary row.

This module has no index access and does not change persistent Record/RID.
It is used by ranked planning; it is not a general-purpose SQL evaluator.
"""

from dataclasses import fields, is_dataclass

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import (
    BoundScoreExpr,
    ColumnRef,
    Expr,
    ScoreExpr,
    SelectColumn,
    StarExpr,
)

Schema = tuple[ColumnDef, ...]

# SQL identifiers cannot contain a NUL character. Never expose this slot via
# SELECT * or use its name to resolve user-provided expressions.
_INTERNAL_SCORE_NAME = "\x00ranked_score"
_SCORE_KINDS = frozenset({"text", "vector"})


def append_score_schema(schema: Schema) -> Schema:
    """Return a temporary schema with one private FLOAT column appended."""
    if any(column.name == _INTERNAL_SCORE_NAME for column in schema):
        raise QueryExecutionError("internal score slot already exists")
    return (*schema, ColumnDef(_INTERNAL_SCORE_NAME, ColumnType.FLOAT))


def _contains_score(value: object) -> bool:
    """Detect nested score references without introducing parser AST visitors."""
    if isinstance(value, (ScoreExpr, BoundScoreExpr)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_score(item) for item in value)
    if is_dataclass(value):
        return any(_contains_score(getattr(value, field.name)) for field in fields(value))
    return False


def bind_score(expr: Expr, *, position: int, kind: str) -> Expr:
    """Bind one standalone SCORE() to the temporary slot.

    Compound score expressions remain unsupported by the current scalar
    evaluator; reject them explicitly instead of silently leaving an
    unbound expression in the query plan.
    """
    if type(position) is not int or position < 0:
        raise QueryExecutionError("score slot position must be a non-negative integer")
    if kind not in _SCORE_KINDS:
        raise QueryExecutionError(f"unsupported score kind {kind!r}")

    if isinstance(expr, ScoreExpr):
        if expr.kind not in ("auto", kind):
            raise QueryExecutionError(
                f"{expr.kind.upper()} score cannot be used in a {kind} search"
            )
        return BoundScoreExpr(position=position, kind=kind)

    if _contains_score(expr):
        raise QueryExecutionError("SCORE() must be a standalone expression")
    return expr


def visible_projections(
    columns: tuple[SelectColumn, ...],
    base_schema: Schema,
    *,
    position: int,
    kind: str,
) -> tuple[SelectColumn, ...]:
    """Expand SELECT * using only base columns and bind SCORE() projections.

    The caller passes the appended score's position in the *temporary* schema.
    All returned projections contain only user-visible output expressions.
    """
    if position != len(base_schema):
        raise QueryExecutionError("score slot must immediately follow base columns")

    base_columns = tuple(SelectColumn(ColumnRef(col.name)) for col in base_schema)
    if not columns:
        return base_columns

    visible: list[SelectColumn] = []
    for offset, selected in enumerate(columns):
        if isinstance(selected.expr, StarExpr):
            if offset != 0 or len(columns) == 1:
                raise QueryExecutionError("invalid mixed SELECT * projection")
            visible.extend(base_columns)
            continue

        if isinstance(selected.expr, ScoreExpr) and selected.alias is not None:
            if any(col.name == selected.alias for col in base_schema):
                raise QueryExecutionError(
                    f"score alias {selected.alias!r} conflicts with a base column"
                )
        expr = bind_score(selected.expr, position=position, kind=kind)
        visible.append(SelectColumn(expr=expr, alias=selected.alias))

    return tuple(visible)
