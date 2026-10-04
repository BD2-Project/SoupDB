"""Query planner: resolves SQL statements into volcano execution plans.

The planner consumes a catalog by *duck typing*: both the persistent
:class:`engine.common.catalog.Catalog` and the tests fake satisfy the same
interface. It expects the injected object to provide:

- ``schema(name)`` returning the table schema and raising
  ``QueryExecutionError`` for unknown tables.
- ``file_org(name)`` returning the table file organization.
- ``indexes(name)`` returning a mapping ``index name -> Index`` (possibly
  empty) with every physical index on the table.
- ``indexes_for(name, column)`` returning the indexes that cover a specific
  column (``column name -> Index`` is NOT the contract: a table may carry
  several differently-structured indexes on the same column).
- ``create_table(name, columns, engine)`` used by the executor for DDL.
- ``create_index(index_name, table, column, index_type)`` for index DDL.
- ``index_location(index_name)`` returning ``(table, column)`` or None.

Access-path selection picks the index best suited to the predicate: a HASH
index is preferred for point lookups and a BTREE index for inclusive ranges.
A spatial index (an :class:`engine.indexes.rtree.RTree`) is picked only for the
radius predicate it can answer, and never for a scalar ``=``/``BETWEEN``
access path, because its key is a location and not a scalar value.
"""

from dataclasses import dataclass
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnType
from engine.indexes.base import Index
from engine.indexes.rtree import RTree
from engine.query.ast import (
    BeginTransactionStatement,
    BetweenExpr,
    ColumnRef,
    CompareExpr,
    CreateIndexStatement,
    CreateTableStatement,
    DeleteStatement,
    DistanceExpr,
    DropIndexStatement,
    DropTableStatement,
    EndTransactionStatement,
    ExplainStatement,
    Expr,
    FunctionExpr,
    InExpr,
    InsertStatement,
    LikeExpr,
    LimitClause,
    Literal,
    LogicalExpr,
    NotExpr,
    OrderByItem,
    PointExpr,
    SelectColumn,
    SelectStatement,
    Statement,
)
from engine.query.evaluator import Schema
from engine.query.operators import (
    Aggregate,
    Distinct,
    Filter,
    IndexLookup,
    IndexRangeScan,
    Limit,
    Operator,
    Project,
    Sort,
    SpatialIndexScan,
    TableScan,
)
from engine.query.spatial_metrics import EUCLIDEAN
from engine.storage.base import FileOrganization
from engine.storage.disk_manager import DiskManager

#: Comparisons a spatial index can serve: both keep every matching row inside
#: the bounding box ``center ± r``. See :func:`_spatial_radius_scan`.
_RADIUS_OPERATORS = ("<", "<=")

#: Index structures accepted by ``CREATE INDEX``: two scalar ones and the
#: spatial R-Tree, which is the only one that requires a POINT column.
_INDEX_TYPES = ("BTREE", "HASH", "RTREE")


@dataclass(frozen=True)
class Plan:
    """Frozen execution plan: the statement plus the volcano tree for SELECT.

    ``root`` and ``output_schema`` are only set for SELECT; DML/DDL plans are
    executed by the executor directly from the statement.
    """

    statement: Statement
    root: Operator | None = None
    output_schema: Schema | None = None


def plan(statement: Statement, catalog: Any) -> Plan:
    """Build the execution plan for a parsed statement."""
    if isinstance(statement, SelectStatement):
        return _plan_select(statement, catalog)
    if isinstance(statement, ExplainStatement):
        return _plan_explain(statement, catalog)
    if isinstance(statement, InsertStatement):
        return _plan_insert(statement, catalog)
    if isinstance(statement, DeleteStatement):
        return _plan_delete(statement, catalog)
    if isinstance(statement, CreateTableStatement):
        return _plan_create(statement, catalog)
    if isinstance(statement, CreateIndexStatement):
        return _plan_create_index(statement, catalog)
    if isinstance(statement, DropTableStatement):
        return _plan_drop_table(statement, catalog)
    if isinstance(statement, DropIndexStatement):
        return _plan_drop_index(statement, catalog)
    if isinstance(statement, (BeginTransactionStatement, EndTransactionStatement)):
        return _plan_transaction_control(statement)
    raise QueryExecutionError(f"unsupported statement {type(statement).__name__}")


def _plan_select(statement: SelectStatement, catalog: Any) -> Plan:
    schema = catalog.schema(statement.table)
    file_org = catalog.file_org(statement.table)
    # Si el contrato del catálogo llega a exponer su DiskManager, los planes
    # reportan I/O real en explain(); sin él las métricas de disco son cero.
    disk_manager = getattr(catalog, "disk_manager", None)
    aggregates = _collect_aggregates(statement)
    grouped = bool(statement.group_by) or bool(aggregates)
    if grouped and not statement.columns:
        raise QueryExecutionError(
            "SELECT * cannot be combined with GROUP BY or aggregate functions"
        )
    root: Operator = _scan_or_index(catalog, statement, file_org, schema, disk_manager)
    if statement.where is not None:
        _validate_columns(statement.where, schema)
        root = Filter(root, statement.where, schema, disk_manager)
    if grouped:
        root = Aggregate(root, statement.group_by, aggregates, disk_manager)
    if statement.order_by:
        aliases = _projection_aliases(statement.columns)
        order_by = _resolve_alias_order_by(statement.order_by, aliases)
        order_by = _resolve_aggregate_order_by(order_by, aggregates)
        for item in order_by:
            _validate_columns(item.expr, root.schema)
        root = Sort(root, order_by, disk_manager=disk_manager)
    if statement.columns:
        selections = _resolve_aggregate_projections(statement.columns, aggregates)
        root = Project(root, selections, disk_manager)
    if statement.distinct:
        root = Distinct(root, None, disk_manager)
    clause: LimitClause | None = statement.limit
    if clause is not None:
        root = Limit(root, _limit_value(clause.limit), _limit_value(clause.offset), disk_manager)
    return Plan(statement=statement, root=root, output_schema=root.schema)


def _plan_explain(statement: ExplainStatement, catalog: Any) -> Plan:
    """Validate the inner statement against the catalog and keep its tree."""
    if statement.statement is None:
        raise QueryExecutionError("EXPLAIN requires an inner statement")
    inner = plan(statement.statement, catalog)
    return Plan(statement=statement, root=inner.root, output_schema=inner.output_schema)


def _plan_insert(statement: InsertStatement, catalog: Any) -> Plan:
    schema = catalog.schema(statement.table)
    if statement.columns:
        expected = {column.name for column in schema}
        if len(statement.columns) != len(schema) or set(statement.columns) != expected:
            raise QueryExecutionError("insert must provide exactly the table columns")
    width = len(statement.columns) if statement.columns else len(schema)
    for row in statement.values:
        if len(row) != width:
            raise QueryExecutionError("insert value count does not match column count")
    return Plan(statement=statement)


def _plan_delete(statement: DeleteStatement, catalog: Any) -> Plan:
    schema = catalog.schema(statement.table)
    catalog.file_org(statement.table)
    if statement.where is not None:
        _validate_columns(statement.where, schema)
    return Plan(statement=statement)


def _plan_create(statement: CreateTableStatement, catalog: Any) -> Plan:
    if statement.engine not in ("HEAP", "SEQUENTIAL"):
        raise QueryExecutionError(
            f"unsupported engine {statement.engine!r}; use HEAP or SEQUENTIAL"
        )
    seen: set[str] = set()
    for column in statement.columns:
        if column.name in seen:
            raise QueryExecutionError(f"duplicate column {column.name!r}")
        seen.add(column.name)
    return Plan(statement=statement)


def _plan_create_index(statement: CreateIndexStatement, catalog: Any) -> Plan:
    if statement.index_type not in _INDEX_TYPES:
        raise QueryExecutionError(
            f"unsupported index type {statement.index_type!r}; use {', '.join(_INDEX_TYPES)}"
        )
    schema = catalog.schema(statement.table)
    if statement.column not in {column.name for column in schema}:
        raise QueryExecutionError(
            f"unknown column {statement.column!r} in table {statement.table!r}"
        )
    if statement.index_type == "RTREE":
        column = next(column for column in schema if column.name == statement.column)
        if column.type_name is not ColumnType.POINT:
            raise QueryExecutionError(
                f"a spatial index requires a POINT column; column {statement.column!r} of "
                f"table {statement.table!r} is {column.type_name.value}"
            )
    return Plan(statement=statement)


def _plan_drop_table(statement: DropTableStatement, catalog: Any) -> Plan:
    if statement.table.startswith("Sys"):
        raise QueryExecutionError(
            f"table name {statement.table!r} is reserved for system tables (Sys*)"
        )
    catalog.schema(statement.table)
    return Plan(statement=statement)


def _plan_drop_index(statement: DropIndexStatement, catalog: Any) -> Plan:
    if catalog.index_location(statement.index_name) is None:
        raise QueryExecutionError(f"unknown index {statement.index_name!r}")
    return Plan(statement=statement)


def _plan_transaction_control(statement: Statement) -> Plan:
    """Accept ``BEGIN``/``END`` without an operator tree or catalog checks.

    The plan only carries the statement: opening and closing transactions is a
    session duty (thread-local state plus undo journal), so there is nothing
    for the planner to validate nor for the operator tree to run.
    """
    return Plan(statement=statement)


def _limit_value(literal: Literal | None) -> int:
    """Integer value of a LIMIT/OFFSET literal; a missing OFFSET means zero."""
    if literal is None:
        return 0
    if type(literal.value) is not int:
        raise QueryExecutionError("LIMIT/OFFSET must be an integer")
    return literal.value


def _scan_or_index(
    catalog: Any,
    statement: SelectStatement,
    file_org: FileOrganization,
    schema: Schema,
    disk_manager: DiskManager | None,
) -> Operator:
    """Pick the access path: index leaf when the WHERE allows it, else scan."""
    where = statement.where
    if where is None:
        return TableScan(file_org, schema, disk_manager)
    if (
        isinstance(where, CompareExpr)
        and where.op == "="
        and isinstance(where.left, ColumnRef)
        and isinstance(where.right, Literal)
    ):
        index = _index_for_column(catalog, statement.table, where.left.name, point=True)
        if index is not None:
            return IndexLookup(index, file_org, where.right.value, schema, disk_manager)
    if (
        isinstance(where, BetweenExpr)
        and isinstance(where.value, ColumnRef)
        and isinstance(where.lo, Literal)
        and isinstance(where.hi, Literal)
    ):
        index = _index_for_column(catalog, statement.table, where.value.name, point=False)
        if index is not None:
            return IndexRangeScan(
                index, file_org, where.lo.value, where.hi.value, schema, disk_manager
            )
    spatial = _spatial_radius_scan(catalog, statement, file_org, schema, disk_manager)
    if spatial is not None:
        return spatial
    return TableScan(file_org, schema, disk_manager)


def is_spatial_index(index: Index) -> bool:
    """Whether ``index`` is one of the spatial (R-Tree) structures.

    Two things follow from the answer, and both are about the *nature* of the
    structure rather than about the predicate it can serve:

    - the planner never picks it for a scalar ``=``/``BETWEEN`` access path,
      because its key is a location, not a scalar value;
    - INSERT/DELETE maintenance skips NULL keys, which have no representation
      as a point (a NULL coordinate is not a location at all).
    """
    return isinstance(index, RTree)


def _spatial_radius_scan(
    catalog: Any,
    statement: SelectStatement,
    file_org: FileOrganization,
    schema: Schema,
    disk_manager: DiskManager | None,
) -> Operator | None:
    """Plan a radius predicate through a spatial index, or return None.

    The recognized shape is ``distance(<point column>, POINT(x, y)) < r`` (the
    inclusive ``<= r`` too, which shares the very same bounding box). When the
    column carries a spatial index and the metric is euclidean, the circle
    ``distance(p, center) < r`` is contained in the axis-aligned box
    ``center ± r``, because ``|p.x - center.x| <= distance`` and likewise for
    ``y``: the box is a SUPERCONSET of the circle, so the index returns every
    candidate row and never misses one. The box is not a circle though, so the
    ``Filter`` planned on top re-evaluates the exact predicate and the result is
    identical to the full scan; what the index saves is reading (and computing
    distances on) the rows outside the box.

    Returns None - leaving the statement on a full scan, exactly as before -
    when any of the following does not hold:

    - the predicate is a radius comparison the box can contain: ``<`` or ``<=``.
      The complement of a radius (``>``/``>=``/``=``/``<>``) would need the
      negated candidate set, which no range query can express, so it stays a
      full scan.
    - the metric is euclidean. A haversine radius is expressed in kilometres
      over a sphere, so its planar bound depends on the latitude of each row:
      the only safe planar bound would be the whole domain. Kept as a full scan.
    - both the centre and the radius are numeric literals (no column or
      expression on either side of the comparison).
    - the radius is not negative: with a negative radius the answer is the empty
      set, and spending an index query to prove it is pointless.
    - the referenced column is a POINT column and has a spatial index.
    """
    where = statement.where
    if not isinstance(where, CompareExpr) or where.op not in _RADIUS_OPERATORS:
        return None
    expression = where.left
    if not isinstance(expression, DistanceExpr) or expression.metric != EUCLIDEAN:
        return None
    if not isinstance(expression.left, ColumnRef) or not isinstance(expression.right, PointExpr):
        return None
    radius = _number_literal(where.right)
    center_x = _number_literal(expression.right.x)
    center_y = _number_literal(expression.right.y)
    if radius is None or radius < 0 or center_x is None or center_y is None:
        return None
    column = expression.left.name
    if _column_type(schema, column) is not ColumnType.POINT:
        return None
    found = _spatial_index_for_column(catalog, statement.table, column)
    if found is None:
        return None
    index_name, index = found
    return SpatialIndexScan(
        index,
        file_org,
        center=(center_x, center_y),
        radius=radius,
        schema=schema,
        index_name=index_name,
        column=column,
        metric=expression.metric,
        disk_manager=disk_manager,
    )


def _spatial_index_for_column(
    catalog: Any,
    table: str,
    column: str,
) -> tuple[str, Index] | None:
    """Name and spatial index covering ``column``, or None if there is none."""
    try:
        candidates = catalog.indexes_for(table, column)
    except (AttributeError, NotImplementedError):
        return None
    for index_name, candidate in candidates.items():
        if is_spatial_index(candidate) and candidate.supports_range:
            return index_name, candidate
    return None


def _number_literal(expr: Expr) -> float | None:
    """Numeric value of a literal expression, or None when it is not one."""
    if not isinstance(expr, Literal) or type(expr.value) not in (int, float):
        return None
    return float(expr.value)


def _column_type(schema: Schema, name: str) -> ColumnType | None:
    """Type of the column ``name``, or None when the schema has no such column."""
    for column in schema:
        if column.name == name:
            return column.type_name
    return None


def _index_for_column(
    catalog: Any,
    table: str,
    column: str,
    *,
    point: bool,
) -> Index | None:
    """Pick the index covering ``column`` best suited to the access pattern."""
    try:
        candidates = catalog.indexes_for(table, column)
    except (AttributeError, NotImplementedError):
        return None
    if not candidates:
        return None
    candidates = {
        index_name: candidate
        for index_name, candidate in candidates.items()
        if not is_spatial_index(candidate)
    }
    if not candidates:
        return None
    if point:
        for candidate in candidates.values():
            if not candidate.supports_range:
                return candidate
        return next(iter(candidates.values()))
    for candidate in candidates.values():
        if candidate.supports_range:
            return candidate
    return None


def _collect_aggregates(statement: SelectStatement) -> tuple[FunctionExpr, ...]:
    found: list[FunctionExpr] = []
    for selection in statement.columns:
        _walk(selection.expr, found)
    for item in statement.order_by:
        _walk(item.expr, found)
    return tuple(dict.fromkeys(found))


def _resolve_aggregate_projections(
    projections: tuple[SelectColumn, ...],
    aggregates: tuple[FunctionExpr, ...],
) -> tuple[SelectColumn, ...]:
    resolved: list[SelectColumn] = []
    for selection in projections:
        expr = selection.expr
        if isinstance(expr, FunctionExpr):
            for position, agg in enumerate(aggregates, start=1):
                if expr == agg:
                    resolved.append(
                        SelectColumn(
                            ColumnRef(f"{agg.name.lower()}_{position}"),
                            selection.alias,
                        )
                    )
                    break
            else:
                resolved.append(selection)
        else:
            resolved.append(selection)
    return tuple(resolved)


def _resolve_aggregate_order_by(
    order_by: tuple[OrderByItem, ...],
    aggregates: tuple[FunctionExpr, ...],
) -> tuple[OrderByItem, ...]:
    """Replace top-level aggregate expressions in ORDER BY with aggregate columns."""
    resolved: list[OrderByItem] = []
    for item in order_by:
        expr = item.expr
        if isinstance(expr, FunctionExpr):
            for position, agg in enumerate(aggregates, start=1):
                if expr == agg:
                    expr = ColumnRef(f"{agg.name.lower()}_{position}")
                    break
        resolved.append(OrderByItem(expr, item.ascending))
    return tuple(resolved)


def _projection_aliases(columns: tuple[SelectColumn, ...]) -> dict[str, Expr]:
    """Map every projection alias to the expression it names.

    Only explicit aliases are registered: an unaliased projection keeps its
    base column name, which ORDER BY already resolves against the input
    schema. When the same alias appears twice the last projection wins.
    """
    return {selection.alias: selection.expr for selection in columns if selection.alias}


def _resolve_alias_order_by(
    order_by: tuple[OrderByItem, ...],
    aliases: dict[str, Expr],
) -> tuple[OrderByItem, ...]:
    """Replace ORDER BY keys naming a projection alias with that expression.

    ``Sort`` is planned *below* ``Project`` so a key may reference a column
    that is not projected (``SELECT nombre FROM t ORDER BY edad``); the alias
    map is therefore not part of the schema the keys are validated against.
    Instead every key is an alias lookup done *before* validation and before
    the ``Sort`` is built, which keeps both behaviours: any expression of the
    projection list is orderable by its alias, whatever its shape (a column,
    an arithmetic or spatial expression, or an aggregate), and non-projected
    base columns keep working.

    Precedence: an alias shadows a base column of the same name, matching SQL,
    where ORDER BY names the *output* column. ``SELECT venue AS anio FROM t
    ORDER BY anio`` therefore sorts by ``venue`` and not by ``t.anio``; when
    the name is neither an alias nor a base column it stays unresolved and
    :func:`_validate_columns` reports ``unknown column``.

    Aliases resolve one level only: a key that names another alias (``SELECT a
    AS p, p + 1 AS q ORDER BY q``) is not a legal projection here and remains
    an unresolved column.
    """
    resolved: list[OrderByItem] = []
    for item in order_by:
        expr = item.expr
        if isinstance(expr, ColumnRef) and expr.name in aliases:
            expr = aliases[expr.name]
        resolved.append(OrderByItem(expr, item.ascending))
    return tuple(resolved)


def _validate_columns(expr: Expr, schema: Schema) -> None:
    """Raise early when an expression references a column missing from schema."""
    if isinstance(expr, ColumnRef):
        for column in schema:
            if column.name == expr.name:
                return
        raise QueryExecutionError(f"unknown column {expr.name!r}")
    for child in _expr_children(expr):
        _validate_columns(child, schema)


def _walk(expr: Expr, found: list[FunctionExpr]) -> None:
    if isinstance(expr, FunctionExpr):
        found.append(expr)
    for child in _expr_children(expr):
        _walk(child, found)


def _expr_children(expr: Expr) -> tuple[Expr, ...]:
    if isinstance(expr, (CompareExpr, LogicalExpr)):
        return (expr.left, expr.right)
    if isinstance(expr, NotExpr):
        return (expr.operand,)
    if isinstance(expr, BetweenExpr):
        return (expr.value, expr.lo, expr.hi)
    if isinstance(expr, InExpr):
        return (expr.value, *expr.items)
    if isinstance(expr, LikeExpr):
        return (expr.value, expr.pattern)
    if isinstance(expr, FunctionExpr) and expr.arg is not None:
        return (expr.arg,)
    if isinstance(expr, PointExpr):
        return (expr.x, expr.y)
    if isinstance(expr, DistanceExpr):
        return (expr.left, expr.right)
    return ()
