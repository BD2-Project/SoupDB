"""Query planner: resolves SQL statements into volcano execution plans.

The planner consumes a catalog by *duck typing*: both the persistent
:class:`engine.common.catalog.Catalog` and the tests fake satisfy the same
interface. It expects the injected object to provide:

- ``schema(name)`` returning the table schema and raising
  ``QueryExecutionError`` for unknown tables.
- ``file_org(name)`` returning the table file organization.
- ``indexes(name)`` returning a mapping ``column name -> Index`` (possibly
  empty) used for physical access-path selection.
- ``create_table(name, columns, engine)`` used by the executor for DDL.
- ``create_index(index_name, table, column, index_type)`` for index DDL.
"""

from dataclasses import dataclass
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.query.ast import (
    BetweenExpr,
    ColumnRef,
    CompareExpr,
    CreateIndexStatement,
    CreateTableStatement,
    DeleteStatement,
    Expr,
    FunctionExpr,
    InExpr,
    InsertStatement,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    OrderByItem,
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
    Operator,
    Project,
    Sort,
    TableScan,
)
from engine.storage.base import FileOrganization
from engine.storage.disk_manager import DiskManager


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
    if isinstance(statement, InsertStatement):
        return _plan_insert(statement, catalog)
    if isinstance(statement, DeleteStatement):
        return _plan_delete(statement, catalog)
    if isinstance(statement, CreateTableStatement):
        return _plan_create(statement, catalog)
    if isinstance(statement, CreateIndexStatement):
        return _plan_create_index(statement, catalog)
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
        order_by = _resolve_aggregate_order_by(statement.order_by, aggregates)
        for item in order_by:
            _validate_columns(item.expr, root.schema)
        root = Sort(root, order_by, disk_manager=disk_manager)
    if statement.columns:
        selections = _resolve_aggregate_projections(statement.columns, aggregates)
        root = Project(root, selections, disk_manager)
    if statement.distinct:
        root = Distinct(root, None, disk_manager)
    return Plan(statement=statement, root=root, output_schema=root.schema)


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
    if statement.index_type not in ("BTREE", "HASH"):
        raise QueryExecutionError(
            f"unsupported index type {statement.index_type!r}; use BTREE or HASH"
        )
    schema = catalog.schema(statement.table)
    if statement.column not in {column.name for column in schema}:
        raise QueryExecutionError(
            f"unknown column {statement.column!r} in table {statement.table!r}"
        )
    return Plan(statement=statement)


def _scan_or_index(
    catalog: Any,
    statement: SelectStatement,
    file_org: FileOrganization,
    schema: Schema,
    disk_manager: DiskManager | None,
) -> Operator:
    """Pick the access path: index leaf when the WHERE allows it, else scan."""
    try:
        index_map = catalog.indexes(statement.table)
    except (AttributeError, NotImplementedError):
        return TableScan(file_org, schema, disk_manager)
    if not index_map:
        return TableScan(file_org, schema, disk_manager)
    where = statement.where
    if where is None:
        return TableScan(file_org, schema, disk_manager)
    if (
        isinstance(where, CompareExpr)
        and where.op == "="
        and isinstance(where.left, ColumnRef)
        and isinstance(where.right, Literal)
    ):
        index = index_map.get(where.left.name)
        if index is not None:
            return IndexLookup(index, file_org.fetch, where.right.value, schema, disk_manager)
    if (
        isinstance(where, BetweenExpr)
        and isinstance(where.value, ColumnRef)
        and isinstance(where.lo, Literal)
        and isinstance(where.hi, Literal)
    ):
        index = index_map.get(where.value.name)
        if index is not None and index.supports_range:
            return IndexRangeScan(
                index, file_org.fetch, where.lo.value, where.hi.value, schema, disk_manager
            )
    return TableScan(file_org, schema, disk_manager)


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
    return ()
