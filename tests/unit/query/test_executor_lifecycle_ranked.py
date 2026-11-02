"""C06: SELECT and EXPLAIN ANALYZE always close their operator root.

The executor owns a query's operator lifecycle even if opening, reading,
explaining or decoding fails. Cleanup errors must not hide the first failure.
"""

import json

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import ExplainStatement, SelectStatement
from engine.query.executor import execute
from engine.query.operators import Operator, PlanNode
from engine.query.planner import Plan

SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("contenido", ColumnType.TEXT),
)
ROWS = ((1, "uno"), (2, "dos"))


class LifecycleProbe(Operator):
    """Deterministic Volcano root with observable calls and scripted errors."""

    def __init__(
        self,
        rows: tuple[Record, ...] | None = None,
        *,
        open_error: Exception | None = None,
        next_error_at: int | None = None,
        close_error: Exception | None = None,
        explain_error: Exception | None = None,
    ) -> None:
        self.schema = SCHEMA
        self.records = (
            rows if rows is not None else tuple(Record(encode_row(row, SCHEMA)) for row in ROWS)
        )
        self.open_error = open_error
        self.next_error_at = next_error_at
        self.close_error = close_error
        self.explain_error = explain_error
        self.events: list[str] = []
        self.position = 0
        self.emitted = 0

    def open(self) -> None:
        self.events.append("open")
        self.position = 0
        self.emitted = 0
        if self.open_error is not None:
            raise self.open_error

    def next(self) -> Record | None:
        self.events.append("next")
        if self.next_error_at == self.position:
            raise RuntimeError("error de lectura original")
        if self.position >= len(self.records):
            return None
        record = self.records[self.position]
        self.position += 1
        self.emitted += 1
        return record

    def close(self) -> None:
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error

    def explain(self) -> PlanNode:
        self.events.append("explain")
        if self.explain_error is not None:
            raise self.explain_error
        return PlanNode(
            op="LifecycleProbe",
            detail={},
            rows=self.emitted,
            elapsed_ms=0.0,
            disk_reads=0,
            disk_writes=0,
        )


def _select_plan(root: LifecycleProbe) -> Plan:
    return Plan(
        statement=SelectStatement(columns=(), table="documentos"),
        root=root,
        output_schema=SCHEMA,
    )


def _explain_plan(root: LifecycleProbe) -> Plan:
    inner = SelectStatement(columns=(), table="documentos")
    return Plan(
        statement=ExplainStatement(
            sql="SELECT * FROM documentos",
            statement=inner,
            analyze=True,
            json_format=True,
        ),
        root=root,
        output_schema=SCHEMA,
    )


def test_select_success_closes_root() -> None:
    root = LifecycleProbe()

    result = execute(_select_plan(root), None)

    assert result.rows == ROWS
    assert root.events == ["open", "next", "next", "next", "close"]


def test_select_empty_result_still_closes_root() -> None:
    root = LifecycleProbe(rows=())

    assert execute(_select_plan(root), None).rows == ()
    assert root.events == ["open", "next", "close"]


def test_select_open_failure_closes_root_and_preserves_error() -> None:
    error = RuntimeError("error de apertura original")
    root = LifecycleProbe(open_error=error)

    with pytest.raises(RuntimeError, match="error de apertura original") as info:
        execute(_select_plan(root), None)

    assert info.value is error
    assert root.events == ["open", "close"]


def test_select_next_failure_closes_root_and_preserves_error() -> None:
    root = LifecycleProbe(next_error_at=1)

    with pytest.raises(RuntimeError, match="error de lectura original"):
        execute(_select_plan(root), None)

    assert root.events == ["open", "next", "next", "close"]


def test_select_decode_failure_closes_root() -> None:
    root = LifecycleProbe(rows=(Record(b"\x01"),))

    with pytest.raises(QueryExecutionError, match="truncated record"):
        execute(_select_plan(root), None)

    assert root.events == ["open", "next", "close"]


def test_select_success_propagates_close_failure() -> None:
    root = LifecycleProbe(close_error=RuntimeError("error de cierre"))

    with pytest.raises(RuntimeError, match="error de cierre"):
        execute(_select_plan(root), None)

    assert root.events[-1] == "close"
    assert root.events.count("close") == 1


def test_select_open_failure_is_not_masked_by_close_failure() -> None:
    error = RuntimeError("error de apertura original")
    root = LifecycleProbe(
        open_error=error,
        close_error=RuntimeError("error de cierre secundario"),
    )

    with pytest.raises(RuntimeError, match="error de apertura original") as info:
        execute(_select_plan(root), None)

    assert info.value is error
    assert root.events == ["open", "close"]


def test_select_next_failure_is_not_masked_by_close_failure() -> None:
    root = LifecycleProbe(
        next_error_at=0,
        close_error=RuntimeError("error de cierre secundario"),
    )

    with pytest.raises(RuntimeError, match="error de lectura original"):
        execute(_select_plan(root), None)

    assert root.events == ["open", "next", "close"]


def test_explain_analyze_success_closes_root() -> None:
    root = LifecycleProbe()

    result = execute(_explain_plan(root), None)

    assert result.columns[0].name == "QUERY PLAN"
    assert json.loads(result.rows[0][0])["op"] == "LifecycleProbe"
    assert json.loads(result.rows[0][0])["rows"] == 2
    assert root.events == ["open", "next", "next", "next", "explain", "close"]


def test_explain_analyze_open_failure_closes_root() -> None:
    root = LifecycleProbe(open_error=RuntimeError("error de apertura original"))

    with pytest.raises(RuntimeError, match="error de apertura original"):
        execute(_explain_plan(root), None)

    assert root.events == ["open", "close"]


def test_explain_analyze_next_failure_explains_then_closes_root() -> None:
    root = LifecycleProbe(next_error_at=1)

    with pytest.raises(RuntimeError, match="error de lectura original"):
        execute(_explain_plan(root), None)

    assert root.events == ["open", "next", "next", "explain", "close"]


def test_explain_analyze_explain_failure_closes_root() -> None:
    root = LifecycleProbe(explain_error=RuntimeError("error de explain"))

    with pytest.raises(RuntimeError, match="error de explain"):
        execute(_explain_plan(root), None)

    assert root.events[-2:] == ["explain", "close"]


def test_explain_analyze_explain_failure_not_masked_by_close_error() -> None:
    error = RuntimeError("error de explain")
    root = LifecycleProbe(
        explain_error=error,
        close_error=RuntimeError("error de cierre secundario"),
    )

    with pytest.raises(RuntimeError, match="error de explain") as info:
        execute(_explain_plan(root), None)

    assert info.value is error
    assert root.events[-2:] == ["explain", "close"]


def test_explain_analyze_success_propagates_close_failure() -> None:
    root = LifecycleProbe(close_error=RuntimeError("error de cierre"))

    with pytest.raises(RuntimeError, match="error de cierre"):
        execute(_explain_plan(root), None)

    assert root.events[-2:] == ["explain", "close"]
    assert root.events.count("close") == 1
