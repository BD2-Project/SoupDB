"""Record serialization / deserialization.

Rows are encoded to a packed byte blob and stored in :class:`Record.data`.
Layout: fixed types use a fixed size (INT 4 bytes, FLOAT 8 bytes, BOOL 1 byte)
and variable-length types (VARCHAR/TEXT) are stored as a ``uint32`` length
prefix followed by the UTF-8 bytes. All integers are little-endian; no
alignment. Empty rows encode to ``b""``.
"""

import struct
from dataclasses import dataclass

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType

_LEN_FORMAT = "<I"
_LEN_SIZE = struct.calcsize(_LEN_FORMAT)


@dataclass
class Record:
    """A record stored as a raw byte blob."""

    data: bytes


def encode_row(row: tuple[object, ...], columns: tuple[ColumnDef, ...]) -> bytes:
    """Serialize ``row`` into bytes using ``columns`` as the layout."""
    if len(row) != len(columns):
        raise QueryExecutionError(f"row has {len(row)} values but schema expects {len(columns)}")
    out = bytearray()
    for value, column in zip(row, columns, strict=True):
        encode_value(out, value, column)
    return bytes(out)


def encode_value(out: bytearray, value: object, column: ColumnDef) -> None:
    """Encode a single column value, appending its bytes to ``out``."""
    if column.type_name is ColumnType.INT:
        if type(value) is not int:
            raise QueryExecutionError(
                f"column {column.name!r} expects INT, got {type(value).__name__}"
            )
        out += value.to_bytes(4, "little")
        return

    if column.type_name is ColumnType.FLOAT:
        if type(value) is not float:
            raise QueryExecutionError(
                f"column {column.name!r} expects FLOAT, got {type(value).__name__}"
            )
        out += struct.pack("<d", value)
        return

    if column.type_name is ColumnType.BOOL:
        if type(value) is not bool:
            raise QueryExecutionError(
                f"column {column.name!r} expects BOOL, got {type(value).__name__}"
            )
        out += b"\x01" if value else b"\x00"
        return

    if column.type_name in (ColumnType.VARCHAR, ColumnType.TEXT):
        encode_text(out, value, column)
        return

    raise QueryExecutionError(f"unsupported column type {column.type_name!r}")


def encode_text(out: bytearray, value: object, column: ColumnDef) -> None:
    if type(value) is not str:
        raise QueryExecutionError(
            f"column {column.name!r} expects text, got {type(value).__name__}"
        )
    if column.type_name is ColumnType.VARCHAR and column.length is not None:
        if len(value) > column.length:
            raise QueryExecutionError(f"column {column.name!r} exceeds VARCHAR({column.length})")
    raw = value.encode("utf-8")
    out += struct.pack(_LEN_FORMAT, len(raw))
    out += raw


def decode_row(data: bytes, columns: tuple[ColumnDef, ...]) -> tuple[object, ...]:
    """Deserialize ``data`` into a row following ``columns``."""
    values = []
    offset = 0
    for column in columns:
        value, offset = decode_value(data, offset, column)
        values.append(value)
    if offset != len(data):
        raise QueryExecutionError(f"trailing bytes after record: got {len(data) - offset} extra")
    return tuple(values)


def decode_value(data: bytes, offset: int, column: ColumnDef) -> tuple[object, int]:
    """Decode one column value from ``data`` at ``offset``; returns value and new offset."""
    if column.type_name is ColumnType.INT:
        value, offset = _take_bytes(data, offset, 4)
        return int.from_bytes(value, "little"), offset

    if column.type_name is ColumnType.FLOAT:
        value, offset = _take_bytes(data, offset, 8)
        return struct.unpack("<d", value)[0], offset

    if column.type_name is ColumnType.BOOL:
        value, offset = _take_bytes(data, offset, 1)
        if value == b"\x00":
            return False, offset
        if value == b"\x01":
            return True, offset
        raise QueryExecutionError("invalid BOOL byte")

    if column.type_name in (ColumnType.VARCHAR, ColumnType.TEXT):
        return decode_text(data, offset)

    raise QueryExecutionError(f"unsupported column type {column.type_name!r}")


def decode_text(data: bytes, offset: int) -> tuple[object, int]:
    raw_length, offset = _take_bytes(data, offset, _LEN_SIZE)
    (length,) = struct.unpack(_LEN_FORMAT, raw_length)
    raw, offset = _take_bytes(data, offset, length)
    try:
        return raw.decode("utf-8"), offset
    except UnicodeDecodeError as exc:
        raise QueryExecutionError("invalid UTF-8 in text column") from exc


def _take_bytes(data: bytes, offset: int, size: int) -> tuple[bytes, int]:
    end = offset + size
    if end > len(data):
        raise QueryExecutionError(
            f"truncated record: need {size} bytes at offset {offset}, have {len(data)}"
        )
    return data[offset:end], end
