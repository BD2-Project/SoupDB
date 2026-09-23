"""Binary wire protocol v1 between the gestor and the rsoup driver.

Frame layout (big-endian):

.. code-block::

    +--------+---------+---------+------------+------------------+
    | magic  | version | opcode  | length     | payload          |
    | 2B "SP"| 1B 0x01 | 1B      | 4B (u32 BE)| (length bytes)   |
    +--------+---------+---------+------------+------------------+

Requests: PING 0x01, BEGIN 0x03, COMMIT 0x04, ROLLBACK 0x05, QUERY 0x06.
Responses: PONG 0x02, OK 0x10, RESULT 0x11, ERROR 0x12.

Payload encodings are documented in ``docs/protocolo.md``; both codecs
(Python here, Rust in ``rsoup/src/protocol.rs``) implement the same contract.
"""

from __future__ import annotations

import socket
import struct
from typing import Any

from engine.common.errors import (
    ContractViolation,
    DeadlockDetected,
    LockNotGranted,
    ProtocolError,
    QueryExecutionError,
    QueryParseError,
    RecordNotFound,
    SoupDBError,
    TransactionError,
    UnsupportedOperation,
)
from engine.common.schema import ColumnDef, ColumnType
from engine.query.resultset import ResultSet

MAGIC = b"SP"
VERSION = 0x01
HEADER_SIZE = 8  # magic(2) + version(1) + opcode(1) + length(4)

# Opcodes de request
OP_PING = 0x01
OP_BEGIN = 0x03
OP_COMMIT = 0x04
OP_ROLLBACK = 0x05
OP_QUERY = 0x06

# Opcodes de response
OP_PONG = 0x02
OP_OK = 0x10
OP_RESULT = 0x11
OP_ERROR = 0x12

# Tags de valor en las celdas de RESULT
TAG_NULL = 0x00
TAG_INT = 0x01
TAG_FLOAT = 0x02
TAG_TEXT = 0x03
TAG_BOOL = 0x04

# Códigos de tipo de columna en RESULT
TYPE_INT = 0x01
TYPE_FLOAT = 0x02
TYPE_VARCHAR = 0x03
TYPE_TEXT = 0x04
TYPE_BOOL = 0x05

# Códigos de error en RESULT/ERROR
ERR_GENERIC = 0x00
ERR_PARSE = 0x01
ERR_EXECUTION = 0x02
ERR_TRANSACTION = 0x03
ERR_LOCK_TIMEOUT = 0x04
ERR_DEADLOCK = 0x05
ERR_RECORD_NOT_FOUND = 0x06
ERR_CONTRACT = 0x07
ERR_UNSUPPORTED = 0x08


def encode_frame(opcode: int, payload: bytes = b"") -> bytes:
    """Encode a full frame: magic + version + opcode + length + payload."""
    return MAGIC + struct.pack(">BBI", VERSION, opcode, len(payload)) + payload


def decode_frame(data: bytes) -> tuple[int, bytes]:
    """Decode a complete frame from a buffer (raises ProtocolError on misuse)."""
    if len(data) < HEADER_SIZE:
        raise ProtocolError(f"frame shorter than header: {len(data)} bytes")
    if data[:2] != MAGIC:
        raise ProtocolError(f"bad magic {data[:2]!r}")
    version, opcode, length = struct.unpack_from(">BBI", data, 2)
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    payload = data[HEADER_SIZE : HEADER_SIZE + length]
    if len(payload) != length:
        raise ProtocolError(f"truncated frame: expected {length} payload bytes")
    return opcode, payload


def recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    """Read exactly one frame from a socket."""
    header = _recv_exact(sock, HEADER_SIZE)
    if header[:2] != MAGIC:
        raise ProtocolError(f"bad magic {header[:2]!r}")
    version, opcode, length = struct.unpack_from(">BBI", header, 2)
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    payload = _recv_exact(sock, length)
    return opcode, payload


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        chunks += chunk
    return bytes(chunks)


# --- Payloads de request ------------------------------------------------


def encode_query(sql: str) -> bytes:
    data = sql.encode("utf-8")
    return struct.pack(">I", len(data)) + data


def decode_query(payload: bytes) -> str:
    (length,) = struct.unpack_from(">I", payload, 0)
    return payload[4 : 4 + length].decode("utf-8")


# --- Payloads de response -----------------------------------------------


def encode_ok(affected: int) -> bytes:
    return struct.pack(">I", affected)


def decode_ok(payload: bytes) -> int:
    (affected,) = struct.unpack(">I", payload)
    return affected


def encode_error(code: int, message: str) -> bytes:
    data = message.encode("utf-8")
    return struct.pack(">BH", code, len(data)) + data


def decode_error(payload: bytes) -> tuple[int, str]:
    code, length = struct.unpack_from(">BH", payload, 0)
    return code, payload[3 : 3 + length].decode("utf-8")


def encode_resultset(result: ResultSet) -> bytes:
    """Serialize a ResultSet into a RESULT payload."""
    out = bytearray()
    out += struct.pack(">H", len(result.columns))
    for column in result.columns:
        name = column.name.encode("utf-8")
        out += struct.pack(">H", len(name)) + name
        out += struct.pack(">BH", _type_code(column.type_name), column.length or 0)
    out += struct.pack(">I", len(result.rows))
    for row in result.rows:
        for value in row:
            _encode_value(out, value)
    return bytes(out)


def decode_resultset(payload: bytes) -> ResultSet:
    columns, offset = _decode_columns(payload, 0)
    (row_count,) = struct.unpack_from(">I", payload, offset)
    offset += 4
    rows: list[tuple[object, ...]] = []
    for _ in range(row_count):
        row: list[object] = []
        for _column in columns:
            value, offset = _decode_value(payload, offset)
            row.append(value)
        rows.append(tuple(row))
    return ResultSet(columns=columns, rows=tuple(rows))


def _decode_columns(payload: bytes, offset: int) -> tuple[tuple[ColumnDef, ...], int]:
    (count,) = struct.unpack_from(">H", payload, offset)
    offset += 2
    columns: list[ColumnDef] = []
    for _ in range(count):
        (name_len,) = struct.unpack_from(">H", payload, offset)
        offset += 2
        name = payload[offset : offset + name_len].decode("utf-8")
        offset += name_len
        (type_code,) = struct.unpack_from(">B", payload, offset)
        offset += 1
        (length,) = struct.unpack_from(">H", payload, offset)
        offset += 2
        columns.append(ColumnDef(name, _column_type(type_code), length or None))
    return tuple(columns), offset


def _encode_value(out: bytearray, value: Any) -> None:
    if value is None:
        out += bytes((TAG_NULL,))
    elif isinstance(value, bool):
        out += bytes((TAG_BOOL, 1 if value else 0))
    elif isinstance(value, int):
        out += struct.pack(">Bi", TAG_INT, value)
    elif isinstance(value, float):
        out += struct.pack(">Bd", TAG_FLOAT, value)
    elif isinstance(value, str):
        data = value.encode("utf-8")
        out += struct.pack(">BH", TAG_TEXT, len(data)) + data
    else:
        raise ProtocolError(f"cannot encode value {value!r}")


def _decode_value(payload: bytes, offset: int) -> tuple[object, int]:
    (tag,) = struct.unpack_from(">B", payload, offset)
    offset += 1
    if tag == TAG_NULL:
        return None, offset
    if tag == TAG_INT:
        (value,) = struct.unpack_from(">i", payload, offset)
        return value, offset + 4
    if tag == TAG_FLOAT:
        (value,) = struct.unpack_from(">d", payload, offset)
        return value, offset + 8
    if tag == TAG_TEXT:
        (length,) = struct.unpack_from(">H", payload, offset)
        offset += 2
        return payload[offset : offset + length].decode("utf-8"), offset + length
    if tag == TAG_BOOL:
        return bool(payload[offset]), offset + 1
    raise ProtocolError(f"unknown value tag {tag}")


def _type_code(type_name: ColumnType) -> int:
    return {
        ColumnType.INT: TYPE_INT,
        ColumnType.FLOAT: TYPE_FLOAT,
        ColumnType.VARCHAR: TYPE_VARCHAR,
        ColumnType.TEXT: TYPE_TEXT,
        ColumnType.BOOL: TYPE_BOOL,
    }[type_name]


def _column_type(code: int) -> ColumnType:
    return {
        TYPE_INT: ColumnType.INT,
        TYPE_FLOAT: ColumnType.FLOAT,
        TYPE_VARCHAR: ColumnType.VARCHAR,
        TYPE_TEXT: ColumnType.TEXT,
        TYPE_BOOL: ColumnType.BOOL,
    }[code]


def error_code(exc: BaseException) -> int:
    """Map an engine exception to a wire error code."""
    if isinstance(exc, QueryParseError):
        return ERR_PARSE
    if isinstance(exc, QueryExecutionError):
        return ERR_EXECUTION
    if isinstance(exc, DeadlockDetected):
        return ERR_DEADLOCK
    if isinstance(exc, LockNotGranted):
        return ERR_LOCK_TIMEOUT
    if isinstance(exc, TransactionError):
        return ERR_TRANSACTION
    if isinstance(exc, RecordNotFound):
        return ERR_RECORD_NOT_FOUND
    if isinstance(exc, ContractViolation):
        return ERR_CONTRACT
    if isinstance(exc, UnsupportedOperation):
        return ERR_UNSUPPORTED
    if isinstance(exc, SoupDBError):
        return ERR_GENERIC
    return ERR_GENERIC
