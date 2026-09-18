"""Volcano operator model.

Frozen contract between query processing and the rest of the engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import (
    BetweenExpr,
    ColumnRef,
    CompareExpr,
    Expr,
    FunctionExpr,
    InExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    SelectColumn,
)
from engine.query.evaluator import Schema, evaluate
from engine.storage.base import FileOrganization
from engine.storage.disk_manager import DiskManager


@dataclass
class PlanNode:
    """Serializable execution plan node (contract with the frontend).

    ``op`` is a free string on purpose: future operators
    (InvertedIndexScan, HNSWSearch) appear without touching the renderer.
    """

    op: str
    detail: dict[str, Any]
    rows: int
    elapsed_ms: float
    disk_reads: int
    disk_writes: int
    children: list["PlanNode"] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Return the JSON contract shape of the plan node."""
        return {
            "op": self.op,
            "detail": self.detail,
            "rows": self.rows,
            "elapsed_ms": self.elapsed_ms,
            "disk_reads": self.disk_reads,
            "disk_writes": self.disk_writes,
            "children": [child.to_json() for child in self.children],
        }


class Operator(ABC):
    """Base interface for all Volcano query operators."""

    @abstractmethod
    def open(self) -> None:
        """Prepare the operator for execution."""

    @abstractmethod
    def next(self) -> Record | None:
        """Return the next record or None when exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the operator."""

    @abstractmethod
    def explain(self) -> PlanNode:
        """Return the execution plan node for this operator."""


class _VolcanoBase(Operator):
    """Shared lifecycle and metrics for concrete operators.

    Concrete operators emit records (``next``) and produce the plan info
    (``explain``); the base tracks rows emitted, elapsed wall-clock time and
    the delta of disk reads/writes of the injected ``DiskManager``.
    """

    def __init__(self, children: tuple["Operator", ...], disk_manager: DiskManager | None) -> None:
        self._children = children
        self._disk = disk_manager
        self._rows = 0
        self._started = False
        self._start = 0.0
        self._reads0 = 0
        self._writes0 = 0
        self.schema: Schema = ()
        """Schema of the records this operator emits."""

    def open(self) -> None:
        self._rows = 0
        self._started = True
        self._start = perf_counter()
        self._reads0 = self._disk.reads if self._disk is not None else 0
        self._writes0 = self._disk.writes if self._disk is not None else 0
        for child in self._children:
            child.open()

    def close(self) -> None:
        self._started = False
        for child in self._children:
            child.close()

    def _plan(self, op: str, detail: dict[str, Any]) -> PlanNode:
        elapsed = (perf_counter() - self._start) * 1000 if self._started else 0.0
        delta_reads = 0
        delta_writes = 0
        if self._disk is not None:
            delta_reads = self._disk.reads - self._reads0
            delta_writes = self._disk.writes - self._writes0
        return PlanNode(
            op=op,
            detail=detail,
            rows=self._rows,
            elapsed_ms=round(elapsed, 3),
            disk_reads=delta_reads,
            disk_writes=delta_writes,
            children=[child.explain() for child in self._children],
        )


class TableScan(_VolcanoBase):
    """Sequential scan of a file organization (leaf operator)."""

    def __init__(
        self,
        file_org: FileOrganization,
        schema: Schema,
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((), disk_manager)
        self._file_org = file_org
        self.schema = schema

    def open(self) -> None:
        super().open()
        self._iterator = self._file_org.scan()

    def next(self) -> Record | None:
        try:
            _, record = next(self._iterator)
        except StopIteration:
            return None
        self._rows += 1
        return record

    def close(self) -> None:
        self._iterator = None
        super().close()

    def explain(self) -> PlanNode:
        columns = [column.name for column in self.schema]
        return self._plan("TableScan", {"columns": columns})


class Filter(_VolcanoBase):
    """Passes through the records that satisfy a predicate."""

    def __init__(
        self,
        child: Operator,
        predicate: Expr,
        schema: Schema,
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self._predicate = predicate
        self.schema = schema

    def next(self) -> Record | None:
        while True:
            record = self._child.next()
            if record is None:
                return None
            row = decode_row(record.data, self.schema)
            if evaluate(self._predicate, row, self.schema):
                self._rows += 1
                return record

    def explain(self) -> PlanNode:
        return self._plan("Filter", {"predicate": str(self._predicate)})


_BOOL_EXPRS = (BetweenExpr, CompareExpr, InExpr, LikeExpr, LogicalExpr, NotExpr)


def _literal_type(value: object) -> ColumnType:
    if type(value) is bool:
        return ColumnType.BOOL
    if type(value) is int:
        return ColumnType.INT
    if type(value) is float:
        return ColumnType.FLOAT
    return ColumnType.TEXT


def _infer_type(expr: Expr, schema: Schema) -> ColumnDef:
    """Best-effort column type for a projected expression."""
    if isinstance(expr, ColumnRef):
        for column in schema:
            if column.name == expr.name:
                return column
        raise QueryExecutionError(f"unknown column {expr.name!r}")
    if isinstance(expr, Literal):
        return ColumnDef("", _literal_type(expr.value))
    if isinstance(expr, _BOOL_EXPRS):
        return ColumnDef("", ColumnType.BOOL)
    if isinstance(expr, FunctionExpr):
        return ColumnDef("", ColumnType.FLOAT)
    return ColumnDef("", ColumnType.TEXT)


def _infer_schema(projections: tuple[SelectColumn, ...], input_schema: Schema) -> Schema:
    """Output schema for a projection list: alias or ``column_<n>`` names."""
    columns = []
    for index, selection in enumerate(projections, start=1):
        name = selection.alias or f"column_{index}"
        column = _infer_type(selection.expr, input_schema)
        columns.append(ColumnDef(name, column.type_name, column.length))
    return tuple(columns)


class Project(_VolcanoBase):
    """Emits one value per projected expression, re-encoded as a record."""

    def __init__(
        self,
        child: Operator,
        projections: tuple[SelectColumn, ...],
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self._projections = projections
        self.schema = _infer_schema(projections, child.schema)

    def next(self) -> Record | None:
        record = self._child.next()
        if record is None:
            return None
        row = decode_row(record.data, self._child.schema)
        values = tuple(
            evaluate(selection.expr, row, self._child.schema) for selection in self._projections
        )
        self._rows += 1
        return Record(data=encode_row(values, self.schema))

    def explain(self) -> PlanNode:
        columns = [column.name for column in self.schema]
        return self._plan("Project", {"columns": columns})


class Distinct(_VolcanoBase):
    """Removes fully duplicated rows, tracking the seen rows in memory."""

    def __init__(
        self,
        child: Operator,
        schema: Schema | None = None,
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self.schema = schema if schema is not None else child.schema

    def open(self) -> None:
        super().open()
        self._seen: set[tuple[object, ...]] = set()

    def next(self) -> Record | None:
        while True:
            record = self._child.next()
            if record is None:
                return None
            row = decode_row(record.data, self.schema)
            if row in self._seen:
                continue
            self._seen.add(row)
            self._rows += 1
            return record

    def explain(self) -> PlanNode:
        columns = [column.name for column in self.schema]
        return self._plan("Distinct", {"columns": columns})
