"""Evaluation of SQL expressions against tuple rows.

The evaluator is a pure module: :func:`evaluate` resolves scalar expressions and
predicates, :func:`evaluate_aggregate` computes aggregate functions over a group
of rows. Rows are plain ``tuple`` of values and the schema is the ordered tuple
of :class:`engine.query.ast.ColumnDef` nodes from the AST (column ``i`` maps to
``row[i]``).

Comparisons are strict: operands must share the exact same type
(``type(a) is type(b)``); an int is never coerced to a float.
"""

import re
from collections.abc import Iterable

from engine.common.errors import QueryExecutionError
from engine.query.ast import (
    BetweenExpr,
    ColumnDef,
    ColumnRef,
    CompareExpr,
    Expr,
    FunctionExpr,
    InExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
)

_COMPARISONS = {"=", "<>", "!=", "<", "<=", ">", ">="}
_AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}

Row = tuple[object, ...]
Schema = tuple[ColumnDef, ...]


def evaluate(expr: Expr, row: Row, schema: Schema) -> object:
    """Evaluate a scalar expression or predicate against a single row."""
    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, ColumnRef):
        return _column_value(expr.name, row, schema)

    if isinstance(expr, CompareExpr):
        left = evaluate(expr.left, row, schema)
        right = evaluate(expr.right, row, schema)
        return _compare(left, expr.op, right)

    if isinstance(expr, LogicalExpr):
        left = _require_bool(evaluate(expr.left, row, schema))
        right = _require_bool(evaluate(expr.right, row, schema))
        return left and right if expr.op == "AND" else left or right

    if isinstance(expr, NotExpr):
        operand = _require_bool(evaluate(expr.operand, row, schema))
        return not operand

    if isinstance(expr, BetweenExpr):
        value = evaluate(expr.value, row, schema)
        lo = evaluate(expr.lo, row, schema)
        hi = evaluate(expr.hi, row, schema)
        _require_same_type(value, lo)
        _require_same_type(value, hi)
        return lo <= value <= hi

    if isinstance(expr, InExpr):
        value = evaluate(expr.value, row, schema)
        items = [evaluate(item, row, schema) for item in expr.items]
        for item in items:
            _require_same_type(value, item)
        return value in items

    if isinstance(expr, LikeExpr):
        value = evaluate(expr.value, row, schema)
        pattern = evaluate(expr.pattern, row, schema)
        return _like(value, pattern)

    if isinstance(expr, FunctionExpr):
        raise QueryExecutionError(f"aggregate {expr.name} is not allowed in a scalar expression")

    raise QueryExecutionError(f"unsupported expression {type(expr).__name__}")


def evaluate_aggregate(expr: FunctionExpr, rows: Iterable[Row], schema: Schema) -> object:
    """Evaluate an aggregate function over a group of rows."""
    name = expr.name.upper()
    if name not in _AGGREGATES:
        raise QueryExecutionError(f"unsupported aggregate {expr.name!r}")
    if expr.arg is None:
        if name != "COUNT":
            raise QueryExecutionError(f"{expr.name} requires a column argument")
        return len(list(rows))

    values = [evaluate(expr.arg, row, schema) for row in rows]
    values = [value for value in values if value is not None]
    if expr.distinct:
        values = list(set(values))

    if name == "COUNT":
        return len(values)
    if name == "SUM":
        return _sum(values)
    if name == "AVG":
        return _avg(values)
    if name == "MIN":
        return _min_or_max(values, "MIN")
    return _min_or_max(values, "MAX")


def _column_value(name: str, row: Row, schema: Schema) -> object:
    for index, column in enumerate(schema):
        if column.name == name:
            if index >= len(row):
                raise QueryExecutionError(f"column {name!r} missing from row")
            return row[index]
    raise QueryExecutionError(f"unknown column {name!r}")


def _require_same_type(left: object, right: object) -> None:
    if type(left) is not type(right):
        raise QueryExecutionError(
            f"cannot mix {type(left).__name__} and {type(right).__name__} values"
        )


def _require_bool(value: object) -> bool:
    if type(value) is not bool:
        raise QueryExecutionError(f"expected boolean operand, got {type(value).__name__}")
    return value


def _compare(left: object, op: str, right: object) -> bool:
    if op not in _COMPARISONS:
        raise QueryExecutionError(f"unsupported comparison operator {op!r}")
    _require_same_type(left, right)
    if op == "=":
        return left == right
    if op in ("<>", "!="):
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left >= right


def _like(value: object, pattern: object) -> bool:
    if type(value) is not str or type(pattern) is not str:
        raise QueryExecutionError("LIKE requires string operands")
    chars = ["^"]
    for char in pattern:
        if char == "%":
            chars.append(".*")
        elif char == "_":
            chars.append(".")
        else:
            chars.append(re.escape(char))
    chars.append("$")
    return re.match("".join(chars), value) is not None


def _is_numeric(value: object) -> bool:
    return type(value) is int or type(value) is float


def _sum(values: list[object]) -> int | float:
    if not values:
        return 0
    if not all(_is_numeric(value) for value in values):
        raise QueryExecutionError("SUM requires numeric values")
    total = 0
    for value in values:
        total += value  # type: ignore[operator]
    return float(total) if any(type(value) is float for value in values) else total


def _avg(values: list[object]) -> float:
    if not values:
        raise QueryExecutionError("AVG of an empty group is undefined")
    if not all(_is_numeric(value) for value in values):
        raise QueryExecutionError("AVG requires numeric values")
    return float(sum(values)) / len(values)  # type: ignore[arg-type]


def _min_or_max(values: list[object], name: str) -> object:
    if not values:
        raise QueryExecutionError(f"{name} of an empty group is undefined")
    first = values[0]
    for value in values[1:]:
        _require_same_type(first, value)
    return min(values) if name == "MIN" else max(values)
