"""Volcano operator model.

Frozen contract between query processing and the rest of the engine.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cmp_to_key
from time import perf_counter
from typing import Any

from engine.algorithms.external_hash import external_hash_group_by
from engine.algorithms.external_sort import external_sort
from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.base import Index, Key
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
    """Sort of the full input stream over one or more keys.

    In-memory by default. When ``memory_limit_bytes`` is given the records are
    sorted externally with ``external_sort`` (bounded memory, temporary runs).
    """

    def __init__(
        self,
        child: Operator,
        order_by: tuple[OrderByItem, ...],
        schema: Schema | None = None,
        disk_manager: DiskManager | None = None,
        memory_limit_bytes: int | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self._order_by = order_by
        self.schema = schema if schema is not None else child.schema
        self._memory_limit = memory_limit_bytes

    def _compare_keys(self, left_keys: tuple[object, ...], right_keys: tuple[object, ...]) -> int:
        for item, left_key, right_key in zip(self._order_by, left_keys, right_keys, strict=True):
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

    def _compare(self, left: object, right: object) -> int:
        return self._compare_keys(left[1], right[1])

    def _compare_records(self, left: Record, right: Record) -> int:
        left_row = decode_row(left.data, self.schema)
        right_row = decode_row(right.data, self.schema)
        left_keys = tuple(evaluate(item.expr, left_row, self.schema) for item in self._order_by)
        right_keys = tuple(evaluate(item.expr, right_row, self.schema) for item in self._order_by)
        return self._compare_keys(left_keys, right_keys)

    def _record_stream(self) -> Iterator[Record]:
        while True:
            record = self._child.next()
            if record is None:
                return
            yield record

    def open(self) -> None:
        super().open()
        if self._memory_limit is not None:
            self._external = external_sort(
                self._record_stream(),
                self._compare_records,
                memory_limit_bytes=self._memory_limit,
            )
            self._buffer = None
        else:
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
        if self._memory_limit is not None:
            try:
                record = next(self._external)
            except StopIteration:
                return None
            self._rows += 1
            return record
        try:
            row, _ = next(self._buffer)
        except StopIteration:
            return None
        self._rows += 1
        return Record(data=encode_row(row, self.schema))

    def close(self) -> None:
        self._buffer = None
        self._external = None
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


@dataclass
class _RunningAggregate:
    """Incremental state of a single aggregate over a group."""

    count: int = 0
    total: int | float = 0
    reference: object | None = None
    minimum: object | None = None
    maximum: object | None = None


class Aggregate(_VolcanoBase):
    """Group-by aggregation over the full input stream.

    In-memory by default. When ``memory_limit_bytes`` is given and there is a
    GROUP BY key while no aggregate uses ``DISTINCT``, the groups are computed
    externally with ``external_hash_group_by`` (bounded memory, disk-backed
    hash partitions). Group keys must be encodable by the external hash
    format: ``int``, ``str`` and tuples of either.
    """

    def __init__(
        self,
        child: Operator,
        group_by: tuple[Expr, ...],
        aggregates: tuple[FunctionExpr, ...],
        disk_manager: DiskManager | None = None,
        memory_limit_bytes: int | None = None,
    ) -> None:
        super().__init__((child,), disk_manager)
        self._child = child
        self._group_by = group_by
        self._aggregates = aggregates
        self._input_schema = child.schema
        self.schema = _aggregate_schema(group_by, aggregates, self._input_schema)
        self._memory_limit = memory_limit_bytes
        self._spill = (
            memory_limit_bytes is not None
            and bool(group_by)
            and not any(expression.distinct for expression in aggregates)
        )
        self._buffer: Iterator[tuple[object, ...]] | None = None
        self._external: Iterator[tuple[object, ...]] | None = None

    def _record_stream(self) -> Iterator[Record]:
        while True:
            record = self._child.next()
            if record is None:
                return
            yield record

    def _group_key(self, record: Record) -> Key:
        row = decode_row(record.data, self._input_schema)
        values = tuple(evaluate(expr, row, self._input_schema) for expr in self._group_by)
        return values[0] if len(values) == 1 else values

    def _aggregate_step(
        self,
        state: list[_RunningAggregate] | None,
        record: Record,
    ) -> list[_RunningAggregate]:
        if state is None:
            state = [_RunningAggregate() for _ in self._aggregates]
        row = decode_row(record.data, self._input_schema)
        for index, expression in enumerate(self._aggregates):
            accumulator = state[index]
            name = expression.name.upper()
            if expression.arg is None:
                if name == "COUNT":
                    accumulator.count += 1
                continue
            value = evaluate(expression.arg, row, self._input_schema)
            if value is None:
                continue
            accumulator.count += 1
            if name in ("SUM", "AVG"):
                if type(value) is not int and type(value) is not float:
                    raise QueryExecutionError(f"{name} requires numeric values")
                accumulator.total += value
            elif name in ("MIN", "MAX"):
                if accumulator.reference is None:
                    accumulator.reference = value
                elif type(accumulator.reference) is not type(value):
                    raise QueryExecutionError(
                        f"cannot mix {type(accumulator.reference).__name__} "
                        f"and {type(value).__name__} values"
                    )
                if name == "MIN":
                    if accumulator.minimum is None or value < accumulator.minimum:
                        accumulator.minimum = value
                else:
                    if accumulator.maximum is None or value > accumulator.maximum:
                        accumulator.maximum = value
        return state

    def _finish_group(self, state: list[_RunningAggregate]) -> tuple[object, ...]:
        values = []
        for index, expression in enumerate(self._aggregates):
            accumulator = state[index]
            name = expression.name.upper()
            if name == "COUNT":
                values.append(accumulator.count)
            elif name == "SUM":
                values.append(accumulator.total)
            elif name == "AVG":
                if accumulator.count == 0:
                    raise QueryExecutionError("AVG of an empty group is undefined")
                values.append(float(accumulator.total) / accumulator.count)
            elif name == "MIN":
                if accumulator.count == 0:
                    raise QueryExecutionError("MIN of an empty group is undefined")
                values.append(accumulator.minimum)
            elif name == "MAX":
                if accumulator.count == 0:
                    raise QueryExecutionError("MAX of an empty group is undefined")
                values.append(accumulator.maximum)
            else:
                raise QueryExecutionError(f"unsupported aggregate {name!r}")
        return tuple(values)

    def _emit_group(self, key: Key, state: list[_RunningAggregate]) -> tuple[object, ...]:
        values = self._finish_group(state)
        if isinstance(key, tuple):
            return key + values
        return (key,) + values

    def open(self) -> None:
        super().open()
        if self._spill:
            groups = external_hash_group_by(
                self._record_stream(),
                self._group_key,
                self._aggregate_step,
                memory_limit_bytes=self._memory_limit,
            )
            self._external = (self._emit_group(key, state) for key, state in groups)
            self._buffer = None
        else:
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
        if self._spill:
            try:
                row = next(self._external)
            except StopIteration:
                return None
            self._rows += 1
            return Record(data=encode_row(row, self.schema))
        try:
            row = next(self._buffer)
        except StopIteration:
            return None
        self._rows += 1
        return Record(data=encode_row(row, self.schema))

    def close(self) -> None:
        self._buffer = None
        self._external = None
        super().close()

    def explain(self) -> PlanNode:
        detail = {
            "group_by": [str(expr) for expr in self._group_by],
            "aggregates": [expr.name for expr in self._aggregates],
        }
        return self._plan("Aggregate", detail)
