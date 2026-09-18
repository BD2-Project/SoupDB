"""Volcano operator model.

Frozen contract between query processing and the rest of the engine.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cmp_to_key
from time import perf_counter
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.base import Index
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
    OrderByItem,
    SelectColumn,
)
from engine.query.evaluator import Schema, evaluate, evaluate_aggregate
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


class IndexLookup(_VolcanoBase):
    """Equality lookup through an index (leaf operator).

    Fetches the candidate RIDs from ``index.search(key)`` and fetches each
    record from the provided callable, skipping stale RIDs whose record no
    longer exists. The caller is responsible for keeping a ``Filter`` on top
    to preserve the semantics of the original predicate.
    """

    def __init__(
        self,
        index: Index,
        fetch: Callable[[RID], Record | None],
        key: object,
        schema: Schema,
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((), disk_manager)
        self._index = index
        self._fetch = fetch
        self._key = key
        self.schema = schema

    def open(self) -> None:
        super().open()
        candidates = self._index.search(self._key)
        self._buffer = iter((rid, self._fetch(rid)) for rid in candidates)

    def next(self) -> Record | None:
        while True:
            try:
                _, record = next(self._buffer)
            except StopIteration:
                return None
            if record is None:
                continue
            self._rows += 1
            return record

    def close(self) -> None:
        self._buffer = None
        super().close()

    def explain(self) -> PlanNode:
        return self._plan("IndexLookup", {"key": str(self._key)})


class IndexRangeScan(_VolcanoBase):
    """Inclusive range equality through an index (leaf operator).

    Requires ``index.supports_range``; the ``Filter`` on top keeps the exact
    boundaries of a non-indexed predicate (e.g. exclusive comparisons).
    """

    def __init__(
        self,
        index: Index,
        fetch: Callable[[RID], Record | None],
        lo: object,
        hi: object,
        schema: Schema,
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((), disk_manager)
        self._index = index
        self._fetch = fetch
        self._lo = lo
        self._hi = hi
        self.schema = schema

    def open(self) -> None:
        super().open()
        if not self._index.supports_range:
            raise QueryExecutionError("range search requested but the index does not support it")
        candidates = self._index.range_search(self._lo, self._hi)
        self._buffer = iter((rid, self._fetch(rid)) for rid in candidates)

    def next(self) -> Record | None:
        while True:
            try:
                _, record = next(self._buffer)
            except StopIteration:
                return None
            if record is None:
                continue
            self._rows += 1
            return record

    def close(self) -> None:
        self._buffer = None
        super().close()

    def explain(self) -> PlanNode:
        detail = {"lo": str(self._lo), "hi": str(self._hi)}
        return self._plan("IndexRangeScan", detail)


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


class Sort(_VolcanoBase):
    """In-memory sort of the full input stream over one or more keys."""

    def __init__(
        self,
        child: Operator,
        order_by: tuple[OrderByItem, ...],
        schema: Schema | None = None,
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self._order_by = order_by
        self.schema = schema if schema is not None else child.schema

    def _compare(self, left: object, right: object) -> int:
        for item, left_key, right_key in zip(self._order_by, left[1], right[1], strict=True):
            if left_key == right_key:
                continue
            try:
                ascending = left_key < right_key
            except TypeError as exc:
                raise QueryExecutionError(
                    f"cannot order by {item.expr}: incompatible types"
                ) from exc
            if ascending is item.ascending:
                return -1
            return 1
        return 0

    def open(self) -> None:
        super().open()
        buffered: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
        while True:
            record = self._child.next()
            if record is None:
                break
            row = decode_row(record.data, self.schema)
            keys = tuple(evaluate(item.expr, row, self.schema) for item in self._order_by)
            buffered.append((row, keys))
        buffered.sort(key=cmp_to_key(self._compare))
        self._buffer = iter(buffered)

    def next(self) -> Record | None:
        try:
            row, _ = next(self._buffer)
        except StopIteration:
            return None
        self._rows += 1
        return Record(data=encode_row(row, self.schema))

    def close(self) -> None:
        self._buffer = None
        super().close()

    def explain(self) -> PlanNode:
        keys = [str(item.expr) for item in self._order_by]
        return self._plan("Sort", {"keys": keys})


def _find_column(name: str, schema: Schema) -> ColumnDef:
    for column in schema:
        if column.name == name:
            return column
    raise QueryExecutionError(f"unknown column {name!r}")


def _aggregate_output_type(expr: FunctionExpr, schema: Schema) -> ColumnType:
    """Output column type for an aggregate function."""
    name = expr.name.upper()
    if name == "COUNT":
        return ColumnType.INT
    if name == "AVG":
        return ColumnType.FLOAT
    if expr.arg is not None:
        column = _infer_type(expr.arg, schema)
        if name in ("MIN", "MAX"):
            return column.type_name
        if name == "SUM":
            return ColumnType.FLOAT if column.type_name is ColumnType.FLOAT else ColumnType.INT
    return ColumnType.INT


def _aggregate_schema(
    group_by: tuple[Expr, ...],
    aggregates: tuple[FunctionExpr, ...],
    schema: Schema,
) -> Schema:
    """Output schema: GROUP BY keys followed by the aggregate results."""
    columns = []
    for index, expr in enumerate(group_by, start=1):
        if isinstance(expr, ColumnRef):
            columns.append(_find_column(expr.name, schema))
            continue
        inferred = _infer_type(expr, schema)
        columns.append(ColumnDef(f"group_{index}", inferred.type_name, inferred.length))
    for index, expr in enumerate(aggregates, start=1):
        columns.append(
            ColumnDef(f"{expr.name.lower()}_{index}", _aggregate_output_type(expr, schema))
        )
    return tuple(columns)


class Aggregate(_VolcanoBase):
    """In-memory group-by aggregation over the full input stream."""

    def __init__(
        self,
        child: Operator,
        group_by: tuple[Expr, ...],
        aggregates: tuple[FunctionExpr, ...],
        disk_manager: DiskManager | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self._group_by = group_by
        self._aggregates = aggregates
        self._input_schema = child.schema
        self.schema = _aggregate_schema(group_by, aggregates, self._input_schema)

    def open(self) -> None:
        super().open()
        groups: dict[tuple[object, ...], list[tuple[object, ...]]]
        if self._group_by:
            groups = {}
        else:
            groups = {(): []}
        while True:
            record = self._child.next()
            if record is None:
                break
            row = decode_row(record.data, self._input_schema)
            key = tuple(evaluate(expr, row, self._input_schema) for expr in self._group_by)
            groups.setdefault(key, []).append(row)
        buffered = []
        for key, group in groups.items():
            values = tuple(
                evaluate_aggregate(expr, group, self._input_schema) for expr in self._aggregates
            )
            buffered.append(key + values)
        self._buffer = iter(buffered)

    def next(self) -> Record | None:
        try:
            row = next(self._buffer)
        except StopIteration:
            return None
        self._rows += 1
        return Record(data=encode_row(row, self.schema))

    def close(self) -> None:
        self._buffer = None
        super().close()

    def explain(self) -> PlanNode:
        detail = {
            "group_by": [str(expr) for expr in self._group_by],
            "aggregates": [expr.name for expr in self._aggregates],
        }
        return self._plan("Aggregate", detail)
