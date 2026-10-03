"""Plan executor that runs volcano plans and DML/DDL statements.

SELECT plans are streamed through the operator tree; INSERT/DELETE/CREATE
statements are executed directly against the catalog (duck-typed, see the
planner contract) and report the affected-row count.
"""

from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.base import Index
from engine.query.ast import (
    CreateIndexStatement,
    CreateTableStatement,
    DeleteStatement,
    DropIndexStatement,
    DropTableStatement,
    ExplainStatement,
    InsertStatement,
    SelectStatement,
    Statement,
)
from engine.query.evaluator import Schema, evaluate
from engine.query.operators import PlanNode
from engine.query.planner import Plan
from engine.query.resultset import ResultSet


def execute(plan: Plan, catalog: Any) -> ResultSet:
    """Execute a frozen plan and return its result set."""
    statement = plan.statement
    if isinstance(statement, SelectStatement):
        return _execute_select(plan)
    if isinstance(statement, InsertStatement):
        return _execute_insert(statement, catalog)
    if isinstance(statement, DeleteStatement):
        return _execute_delete(statement, catalog)
    if isinstance(statement, CreateTableStatement):
        catalog.create_table(statement.table, statement.columns, statement.engine)
        return ResultSet(columns=())
    if isinstance(statement, CreateIndexStatement):
        catalog.create_index(
            statement.index_name,
            statement.table,
            statement.column,
            statement.index_type,
        )
        return ResultSet(columns=())
    if isinstance(statement, DropTableStatement):
        catalog.drop_table(statement.table)
        return ResultSet(columns=())
    if isinstance(statement, DropIndexStatement):
        catalog.drop_index(statement.index_name)
        return ResultSet(columns=())
    if isinstance(statement, ExplainStatement):
        return _execute_explain(plan, catalog)
    raise QueryExecutionError(f"unsupported statement {type(statement).__name__}")


def _execute_select(plan: Plan) -> ResultSet:
    root = plan.root
    if root is None:
        raise QueryExecutionError("select plan has no operator tree")
    root.open()
    try:
        rows = []
        while True:
            record = root.next()
            if record is None:
                break
            rows.append(decode_row(record.data, root.schema))
    finally:
        root.close()
    return ResultSet(columns=plan.output_schema, rows=tuple(rows))


def _execute_insert(statement: InsertStatement, catalog: Any) -> ResultSet:
    schema = catalog.schema(statement.table)
    file_org = catalog.file_org(statement.table)
    columns = statement.columns or tuple(column.name for column in schema)
    positions = [_column_index(schema, name) for name in columns]
    affected = 0
    for row_exprs in statement.values:
        values: list[object] = [None] * len(schema)
        for position, expr in zip(positions, row_exprs, strict=True):
            values[position] = evaluate(expr, (), ())
        rid = file_org.insert(Record(data=encode_row(tuple(values), schema)))
        for position, column in enumerate(schema):
            for index in _indexes_for_column(catalog, statement.table, column.name).values():
                index.insert(values[position], rid)
        affected += 1
    return ResultSet(columns=(), affected=affected)


def _execute_delete(statement: DeleteStatement, catalog: Any) -> ResultSet:
    schema = catalog.schema(statement.table)
    file_org = catalog.file_org(statement.table)
    rows_to_remove = []
    for rid, record in file_org.scan():
        row = decode_row(record.data, schema)
        if statement.where is None or evaluate(statement.where, row, schema):
            rows_to_remove.append((rid, row))
    affected = 0
    for rid, row in rows_to_remove:
        if not file_org.remove(rid):
            continue
        for position, column in enumerate(schema):
            for index in _indexes_for_column(catalog, statement.table, column.name).values():
                index.remove(row[position], rid)
        affected += 1
    return ResultSet(columns=(), affected=affected)


def _indexes_for_column(catalog: Any, table: str, column: str) -> dict[str, Index]:
    """Indexes covering a column, tolerating catalogs without the method."""
    try:
        return catalog.indexes_for(table, column)
    except (AttributeError, NotImplementedError):
        return {}


def _column_index(schema: Schema, name: str) -> int:
    for index, column in enumerate(schema):
        if column.name == name:
            return index
    raise QueryExecutionError(f"unknown column {name!r}")


def _execute_explain(plan: Plan, catalog: Any) -> ResultSet:
    """Describe an execution plan, optionally running it first.

    Plain ``EXPLAIN`` renders the operator tree without executing it, so every
    metric is zero. ``EXPLAIN ANALYZE`` drains the tree (or runs DML/DDL) and
    reports the real rows, wall-clock time and disk I/O per node.
    """
    statement = plan.statement
    if not isinstance(statement, ExplainStatement):
        raise QueryExecutionError("explain executor requires an ExplainStatement")
    if statement.statement is None:
        raise QueryExecutionError("EXPLAIN requires an inner statement")
    if statement.analyze:
        node = _explain_analyze(statement, plan, catalog)
    else:
        node = _explain_describe(plan)
    lines: list[str] = []
    _render_plan(node, 0, lines)
    columns = (ColumnDef("QUERY PLAN", ColumnType.TEXT),)
    return ResultSet(columns=columns, rows=tuple((line,) for line in lines))


def _explain_describe(plan: Plan) -> PlanNode:
    """Render the operator tree without executing it (all metrics zero)."""
    if plan.root is not None:
        return plan.root.explain()
    statement = plan.statement
    if not isinstance(statement, ExplainStatement) or statement.statement is None:
        return _statement_node(statement)
    return _statement_node(statement.statement)


def _explain_analyze(statement: ExplainStatement, plan: Plan, catalog: Any) -> PlanNode:
    """Execute the inner statement, then render the plan with real metrics."""
    inner = statement.statement
    if inner is None:
        raise QueryExecutionError("EXPLAIN requires an inner statement")
    if plan.root is not None:
        plan.root.open()
        try:
            try:
                while plan.root.next() is not None:
                    pass
            finally:
                node = plan.root.explain()
        finally:
            plan.root.close()
        return node
    result = execute(Plan(statement=inner), catalog)
    node = _statement_node(inner)
    node.rows = result.affected
    node.elapsed_ms = 0.0
    return node


def _statement_node(statement: Statement) -> PlanNode:
    """Leaf plan node describing a statement executed directly by the executor."""
    op = type(statement).__name__.removesuffix("Statement")
    detail: dict[str, Any] = {}
    table = getattr(statement, "table", None)
    if table is not None:
        detail["table"] = table
    return PlanNode(op=op, detail=detail, rows=0, elapsed_ms=0.0, disk_reads=0, disk_writes=0)


def _render_plan(node: PlanNode, depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    detail = f" {node.detail}" if node.detail else ""
    lines.append(
        f"{indent}-> {node.op} rows={node.rows} elapsed_ms={node.elapsed_ms} "
        f"disk_reads={node.disk_reads} disk_writes={node.disk_writes}{detail}"
    )
    for child in node.children:
        _render_plan(child, depth + 1, lines)
