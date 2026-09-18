"""Plan executor that runs volcano plans and DML/DDL statements.

SELECT plans are streamed through the operator tree; INSERT/DELETE/CREATE
statements are executed directly against the catalog (duck-typed, see the
planner contract) and report the affected-row count.
"""

from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.query.ast import (
    CreateTableStatement,
    DeleteStatement,
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
        catalog.create_table(statement.table, statement.columns)
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
    columns = statement.columns or tuple(column.name for column in schema)
    positions = [_column_index(schema, name) for name in columns]
    affected = 0
    for row_exprs in statement.values:
        values: list[object] = [None] * len(schema)
        for position, expr in zip(positions, row_exprs, strict=True):
            values[position] = evaluate(expr, (), ())
        file_org.insert(Record(data=encode_row(tuple(values), schema)))
        affected += 1
    return ResultSet(columns=(), affected=affected)


def _execute_delete(statement: DeleteStatement, catalog: Any) -> ResultSet:
    schema = catalog.schema(statement.table)
    file_org = catalog.file_org(statement.table)
    rids_to_remove = []
    for rid, record in file_org.scan():
        if statement.where is None:
            rids_to_remove.append(rid)
            continue
        row = decode_row(record.data, schema)
        if evaluate(statement.where, row, schema):
            rids_to_remove.append(rid)
    affected = 0
    for rid in rids_to_remove:
        if file_org.remove(rid):
            affected += 1
    return ResultSet(columns=(), affected=affected)


def _column_index(schema: Schema, name: str) -> int:
    for index, column in enumerate(schema):
        if column.name == name:
            return index
    raise QueryExecutionError(f"unknown column {name!r}")
