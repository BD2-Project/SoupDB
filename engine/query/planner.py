"""Query planner: resolves SQL statements into volcano execution plans.

The planner consumes a catalog by *duck typing* because the formal contract
of ``engine/common/catalog.py`` is still being negotiated with the team. It
expects the injected object to provide:

- ``schema(name)`` returning the table schema and raising
  ``QueryExecutionError`` for unknown tables.
- ``file_org(name)`` returning the table file organization.
- ``indexes(name)`` returning a mapping ``column name -> Index`` (possibly
  empty) used for physical access-path selection.
- ``create_table(name, columns)`` used by the executor for DDL.

The real catalog replaces this duck-typed one once its contract exists.
"""

from dataclasses import dataclass
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.query.ast import (
    BetweenExpr,
    ColumnRef,
    CompareExpr,
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
    raise QueryExecutionError(f"unsupported statement {type(statement).__name__}")


def _plan_select(statement: SelectStatement, catalog: Any) -> Plan:
    schema = catalog.schema(statement.table)
    file_org = catalog.file_org(statement.table)
    aggregates = _collect_aggregates(statement)
    grouped = bool(statement.group_by) or bool(aggregates)
    if grouped and not statement.columns:
        raise QueryExecutionError(
            "SELECT * cannot be combined with GROUP BY or aggregate functions"
        )
    root: Operator = _scan_or_index(catalog, statement, file_org, schema)
    if statement.where is not None:
        root = Filter(root, statement.where, schema)
    if grouped:
        root = Aggregate(root, statement.group_by, aggregates)
    if statement.order_by:
        root = Sort(root, statement.order_by)
    if statement.columns:
        selections = _resolve_aggregate_projections(statement.columns, aggregates)
        root = Project(root, selections)
    if statement.distinct:
        root = Distinct(root)
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
    catalog.schema(statement.table)
    catalog.file_org(statement.table)
    return Plan(statement=statement)


def _plan_create(statement: CreateTableStatement, catalog: Any) -> Plan:
    seen: set[str] = set()
    for column in statement.columns:
        if column.name in seen:
            raise QueryExecutionError(f"duplicate column {column.name!r}")
        seen.add(column.name)
    return Plan(statement=statement)


def _scan_or_index(
    catalog: Any,
    statement: SelectStatement,
    file_org: FileOrganization,
    schema: Schema,
) -> Operator:
    """Pick the access path: index leaf when the WHERE allows it, else scan."""
    try:
        index_map = catalog.indexes(statement.table)
    except (AttributeError, NotImplementedError):
        return TableScan(file_org, schema)
    if not index_map:
        return TableScan(file_org, schema)
    where = statement.where
    if where is None:
        return TableScan(file_org, schema)
    if (
        isinstance(where, CompareExpr)
        and where.op == "="
        and isinstance(where.left, ColumnRef)
        and isinstance(where.right, Literal)
    ):
        index = index_map.get(where.left.name)
        if index is not None:
            return IndexLookup(index, file_org.fetch, where.right.value, schema)
    if (
        isinstance(where, BetweenExpr)
        and isinstance(where.value, ColumnRef)
        and isinstance(where.lo, Literal)
        and isinstance(where.hi, Literal)
    ):
        index = index_map.get(where.value.name)
        if index is not None and index.supports_range:
            return IndexRangeScan(index, file_org.fetch, where.lo.value, where.hi.value, schema)
    return TableScan(file_org, schema)


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
