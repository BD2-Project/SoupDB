"""Plan executor that runs volcano plans and DML/DDL statements.

SELECT plans are streamed through the operator tree; INSERT/DELETE/CREATE
statements are executed directly against the catalog (duck-typed, see the
planner contract) and report the affected-row count.
"""

from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.indexes.base import Index
from engine.query.ast import (
    CreateIndexStatement,
    CreateTableStatement,
    DeleteStatement,
    DropIndexStatement,
    DropTableStatement,
    InsertStatement,
    SelectStatement,
)
from engine.query.evaluator import Schema, evaluate
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
    try:
        indexes: dict[str, Index] = catalog.indexes(statement.table)
    except (AttributeError, NotImplementedError):
        indexes = {}
    columns = statement.columns or tuple(column.name for column in schema)
    positions = [_column_index(schema, name) for name in columns]
    affected = 0
    for row_exprs in statement.values:
        values: list[object] = [None] * len(schema)
        for position, expr in zip(positions, row_exprs, strict=True):
            values[position] = evaluate(expr, (), ())
        rid = file_org.insert(Record(data=encode_row(tuple(values), schema)))
        for position, column in enumerate(schema):
            index = indexes.get(column.name)
            if index is not None:
                index.insert(values[position], rid)
        affected += 1
    return ResultSet(columns=(), affected=affected)


def _execute_delete(statement: DeleteStatement, catalog: Any) -> ResultSet:
    schema = catalog.schema(statement.table)
    file_org = catalog.file_org(statement.table)
    try:
        indexes: dict[str, Index] = catalog.indexes(statement.table)
    except (AttributeError, NotImplementedError):
        indexes = {}
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
            index = indexes.get(column.name)
            if index is not None:
                index.remove(row[position], rid)
        affected += 1
    return ResultSet(columns=(), affected=affected)


def _column_index(schema: Schema, name: str) -> int:
    for index, column in enumerate(schema):
        if column.name == name:
            return index
    raise QueryExecutionError(f"unknown column {name!r}")
