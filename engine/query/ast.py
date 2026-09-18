"""SQL abstract syntax tree nodes for query processing."""

from dataclasses import dataclass

from engine.common.schema import ColumnDef, ColumnType  # noqa: F401  (re-export)


class Statement:
    """Base class for every SQL statement (SELECT, INSERT, DELETE, CREATE)."""


class Expr:
    """Base class for every SQL expression.

    Expressions appear in projections, WHERE clauses, GROUP BY and
    ORDER BY items.
    """


@dataclass(frozen=True)
class Literal(Expr):
    """A literal value: number, string or boolean."""

    value: int | float | str | bool | None


@dataclass(frozen=True)
class ColumnRef(Expr):
    """A reference to a column by name."""

    name: str


@dataclass(frozen=True)
class CompareExpr(Expr):
    """Comparison ``left OP right`` with OP in ``= <> < <= > >=``."""

    left: Expr
    op: str
    right: Expr


@dataclass(frozen=True)
class LogicalExpr(Expr):
    """Boolean combination ``left AND right`` or ``left OR right``."""

    left: Expr
    op: str
    right: Expr


@dataclass(frozen=True)
class NotExpr(Expr):
    """Negation of a boolean expression."""

    operand: Expr


@dataclass(frozen=True)
class BetweenExpr(Expr):
    """Inclusive range check ``value BETWEEN lo AND hi``."""

    value: Expr
    lo: Expr
    hi: Expr


@dataclass(frozen=True)
class InExpr(Expr):
    """Membership check ``value IN (item, ...)``."""

    value: Expr
    items: tuple[Expr, ...]


@dataclass(frozen=True)
class LikeExpr(Expr):
    """Pattern match ``value LIKE pattern`` supporting ``%`` and ``_``."""

    value: Expr
    pattern: Expr


@dataclass(frozen=True)
class FunctionExpr(Expr):
    """Aggregate or function call such as ``COUNT(*)`` or ``SUM(anio)``.

    ``arg`` is None when the function takes no argument (e.g. ``COUNT(*)``).
    """

    name: str
    arg: Expr | None
    distinct: bool = False


@dataclass(frozen=True)
class SelectColumn:
    """One projected column in a SELECT list."""

    expr: Expr
    alias: str | None = None


@dataclass(frozen=True)
class OrderByItem:
    """One ORDER BY key and its direction."""

    expr: Expr
    ascending: bool = True


@dataclass(frozen=True)
class SelectStatement(Statement):
    """A SELECT query with optional WHERE, GROUP BY and ORDER BY clauses."""

    columns: tuple[SelectColumn, ...]
    table: str
    where: Expr | None = None
    group_by: tuple[Expr, ...] = ()
    order_by: tuple[OrderByItem, ...] = ()
    distinct: bool = False


@dataclass(frozen=True)
class InsertStatement(Statement):
    """``INSERT INTO table (cols) VALUES (...), (...)``."""

    table: str
    columns: tuple[str, ...]
    values: tuple[tuple[Expr, ...], ...]


@dataclass(frozen=True)
class DeleteStatement(Statement):
    """``DELETE FROM table WHERE condition``."""

    table: str
    where: Expr | None


@dataclass(frozen=True)
class CreateTableStatement(Statement):
    """``CREATE TABLE name (col type, ...)``."""

    table: str
    columns: tuple[ColumnDef, ...]
