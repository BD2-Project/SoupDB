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
class BinaryExpr(Expr):
    """Arithmetic operation ``left OP right`` with OP in ``+ - * / %``."""

    left: Expr
    op: str
    right: Expr


@dataclass(frozen=True)
class IsNullExpr(Expr):
    """Null check ``value IS [NOT] NULL``."""

    value: Expr
    negated: bool = False


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
    """``arg`` is None when the function takes no argument (e.g. ``COUNT(*)``)."""

    name: str
    arg: Expr | None
    distinct: bool = False


@dataclass(frozen=True)
class PointExpr(Expr):
    """Spatial constructor ``POINT(x, y)``."""

    x: Expr
    y: Expr


@dataclass(frozen=True)
class DistanceExpr(Expr):
    """``distance(left, right)`` or ``distance(left, right, 'metric')``.

    ``metric`` names the distance metric as a string literal and defaults to
    ``'euclidean'``: ``'euclidean'`` is the plane distance in coordinate units
    and ``'haversine'`` the geodesic distance in kilometres (see
    :mod:`engine.query.spatial_metrics`). The parser stores the canonical
    lowercase name, so ``DistanceExpr`` with two arguments stays equal to the
    same node built explicitly with ``metric='euclidean'``.
    """

    left: Expr
    right: Expr
    metric: str = "euclidean"


@dataclass(frozen=True)
class PolygonExpr(Expr):
    """Spatial constructor ``POLYGON((x1, y1), (x2, y2), ..., (xn, yn))``.

    Vertices are stored as expressions so they may be evaluated per row or be
    literals.
    """

    vertices: tuple[Expr, ...]


@dataclass(frozen=True)
class IntersectsExpr(Expr):
    """Spatial predicate ``intersects(geom_a, geom_b)``.

    Both arguments are geometry expressions (e.g. POINT, POLYGON).
    """

    left: Expr
    right: Expr


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
class HavingClause:
    """Filter applied to grouped rows, ``HAVING expr``."""

    expr: Expr


@dataclass(frozen=True)
class LimitClause:
    """``LIMIT n`` with optional ``OFFSET m``."""

    limit: Literal
    offset: Literal | None = None


@dataclass(frozen=True)
class JoinClause:
    """A JOIN against ``table`` on predicate ``on``."""

    table: str
    on: Expr


@dataclass(frozen=True)
class SelectStatement(Statement):
    """A SELECT query with optional WHERE, GROUP BY, HAVING, ORDER BY,
    LIMIT and JOIN clauses."""

    columns: tuple[SelectColumn, ...]
    table: str
    where: Expr | None = None
    group_by: tuple[Expr, ...] = ()
    order_by: tuple[OrderByItem, ...] = ()
    distinct: bool = False
    having: HavingClause | None = None
    limit: LimitClause | None = None
    joins: tuple[JoinClause, ...] = ()


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
class UpdateStatement(Statement):
    """``UPDATE table SET col = expr, ... [WHERE condition]``."""

    table: str
    assignments: tuple[tuple[str, Expr], ...]
    where: Expr | None = None


@dataclass(frozen=True)
class ExplainStatement(Statement):
    """``EXPLAIN <inner statement>``.

    ``sql`` keeps the raw inner text, ``statement`` its parsed AST when
    available, and ``analyze`` whether the plan must be executed to gather
    real runtime metrics (``EXPLAIN ANALYZE``).

    ``json_format`` lo pide ``EXPLAIN (FORMAT JSON)``: devuelve el árbol
    serializado en una sola fila en vez del dibujo de texto. Es lo que necesita
    el panel de plan del frontend, que no puede reconstruir un árbol a partir
    de líneas indentadas.
    """

    sql: str
    statement: Statement | None = None
    analyze: bool = False
    json_format: bool = False


@dataclass(frozen=True)
class CreateTableStatement(Statement):
    """``CREATE TABLE name (col type [PRIMARY KEY], ...) [ENGINE strategy]``.

    ``engine`` selects the storage strategy: "HEAP" or "SEQUENTIAL", defaulting
    to "HEAP". The planner validates the value before execution.

    ``primary_key`` nombra la columna declarada como clave primaria, si la hay.
    No es una anotación decorativa: el ejecutor crea sobre ella un índice B+,
    que es lo que hace que una búsqueda por clave no recorra la tabla entera.
    """

    table: str
    columns: tuple[ColumnDef, ...]
    engine: str = "HEAP"
    primary_key: str | None = None


@dataclass(frozen=True)
class CreateIndexStatement(Statement):
    """``CREATE INDEX name ON table (column) [TYPE BTREE|HASH]``.

    ``index_type`` defaults to "BTREE".
    """

    index_name: str
    table: str
    column: str
    index_type: str = "BTREE"


@dataclass(frozen=True)
class DropTableStatement(Statement):
    """``DROP TABLE name`` removes a table and its indexes."""

    table: str


@dataclass(frozen=True)
class DropIndexStatement(Statement):
    """``DROP INDEX name`` removes an index."""

    index_name: str


@dataclass(frozen=True)
class BeginTransactionStatement(Statement):
    """``BEGIN [TRANSACTION]``: opens a transaction on the current session thread.

    ``TRANSACTION`` is optional, so both ``BEGIN`` and ``BEGIN TRANSACTION``
    produce this node. The node carries no payload: the transaction itself
    lives in the transactional session (thread-local), not in the AST.
    """


@dataclass(frozen=True)
class EndTransactionStatement(Statement):
    """``END [TRANSACTION]``: closes the active transaction with a commit.

    ``TRANSACTION`` is optional, so both ``END`` and ``END TRANSACTION``
    produce this node. ``END`` closes the transaction committing it, the same
    semantics already supported by ``TransactionalSession.commit()``.
    """
