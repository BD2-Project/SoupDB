"""Codec del protocolo binario TCP de SoupDB (v1 y v2)."""

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
#: Wire versions understood by this server. ``VERSION`` is the current one;
#: ``VERSION_V1`` is kept for drivers that still use numeric type codes.
VERSION_V1 = 0x01
VERSION = 0x02
SUPPORTED_VERSIONS = frozenset({VERSION_V1, VERSION})
HEADER_SIZE = 8  # magic(2) + version(1) + opcode(1) + length(4)

# Opcodes de request
OP_PING = 0x01
OP_BEGIN = 0x03
OP_COMMIT = 0x04
OP_ROLLBACK = 0x05
OP_QUERY = 0x06
OP_PREPARE = 0x07
OP_EXECUTE = 0x08
OP_DEALLOCATE = 0x09
#: Descarga los bytes de un documento multimedia por su referencia (carpeta global).
OP_FETCH = 0x0A

# Opcodes de response
OP_PONG = 0x02
OP_OK = 0x10
OP_RESULT = 0x11
OP_ERROR = 0x12
#: Respuesta a ``OP_FETCH``: media type + bytes del documento.
OP_DOCUMENT = 0x13

# Tags de valor en las celdas de RESULT
TAG_NULL = 0x00
TAG_INT = 0x01
TAG_FLOAT = 0x02
TAG_TEXT = 0x03
TAG_BOOL = 0x04
TAG_POINT = 0x05
TAG_BYTES = 0x06

# Códigos de tipo de columna en RESULT
TYPE_INT = 0x01
TYPE_FLOAT = 0x02
TYPE_VARCHAR = 0x03
TYPE_TEXT = 0x04
TYPE_BOOL = 0x05
TYPE_POINT = 0x06

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


def encode_frame(opcode: int, payload: bytes = b"", *, version: int = VERSION) -> bytes:
    """Encode a full frame: magic + version + opcode + length + payload."""
    return encode_frame_header(opcode, len(payload), version=version) + payload


def encode_frame_header(opcode: int, payload_length: int, *, version: int = VERSION) -> bytes:
    """Encode solo el header de un frame cuyo payload se envía en streaming."""
    if version not in SUPPORTED_VERSIONS:
        raise ProtocolError(f"unsupported protocol version {version}")
    if payload_length > 0xFFFFFFFF:
        raise ProtocolError("frame payload is larger than the u32 length field")
    return MAGIC + struct.pack(">BBI", version, opcode, payload_length)


def decode_frame(data: bytes) -> tuple[int, bytes]:
    """Decode a complete frame from a buffer (raises ProtocolError on misuse)."""
    if len(data) < HEADER_SIZE:
        raise ProtocolError(f"frame shorter than header: {len(data)} bytes")
    if data[:2] != MAGIC:
        raise ProtocolError(f"bad magic {data[:2]!r}")
    version, opcode, length = struct.unpack_from(">BBI", data, 2)
    if version not in SUPPORTED_VERSIONS:
        raise ProtocolError(f"unsupported protocol version {version}")
    payload = data[HEADER_SIZE : HEADER_SIZE + length]
    if len(payload) != length:
        raise ProtocolError(f"truncated frame: expected {length} payload bytes")
    return opcode, payload


def recv_frame_versioned(sock: socket.socket) -> tuple[int, int, bytes]:
    """Read one frame and also report its wire version for the response."""
    header = _recv_exact(sock, HEADER_SIZE)
    if header[:2] != MAGIC:
        raise ProtocolError(f"bad magic {header[:2]!r}")
    version, opcode, length = struct.unpack_from(">BBI", header, 2)
    if version not in SUPPORTED_VERSIONS:
        raise ProtocolError(f"unsupported protocol version {version}")
    payload = _recv_exact(sock, length)
    return version, opcode, payload


def recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    """Read exactly one frame from a socket (version-agnostic)."""
    _version, opcode, payload = recv_frame_versioned(sock)
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


def _encode_name(name: str) -> bytes:
    data = name.encode("utf-8")
    return struct.pack(">H", len(data)) + data


def _decode_name(payload: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from(">H", payload, offset)
    offset += 2
    return payload[offset : offset + length].decode("utf-8"), offset + length


def encode_prepare(name: str, sql: str) -> bytes:
    data = sql.encode("utf-8")
    return _encode_name(name) + struct.pack(">I", len(data)) + data


def decode_prepare(payload: bytes) -> tuple[str, str]:
    name, offset = _decode_name(payload, 0)
    (length,) = struct.unpack_from(">I", payload, offset)
    offset += 4
    return name, payload[offset : offset + length].decode("utf-8")


def encode_execute(name: str, params: tuple[object, ...] = ()) -> bytes:
    out = bytearray(_encode_name(name))
    out += struct.pack(">H", len(params))
    for value in params:
        _encode_value(out, value)
    return bytes(out)


def decode_execute(payload: bytes) -> tuple[str, tuple[object, ...]]:
    name, offset = _decode_name(payload, 0)
    (count,) = struct.unpack_from(">H", payload, offset)
    offset += 2
    params: list[object] = []
    for _ in range(count):
        value, offset = _decode_value(payload, offset)
        params.append(value)
    return name, tuple(params)


def encode_deallocate(name: str) -> bytes:
    return _encode_name(name)


def decode_deallocate(payload: bytes) -> str:
    return _decode_name(payload, 0)[0]


def encode_fetch(reference: str) -> bytes:
    """Payload de ``OP_FETCH``: ``[ref_len u32][reference UTF-8]``."""
    data = reference.encode("utf-8")
    return struct.pack(">I", len(data)) + data


def decode_fetch(payload: bytes) -> str:
    if len(payload) < 4:
        raise ProtocolError("FETCH payload shorter than its length prefix")
    (length,) = struct.unpack_from(">I", payload, 0)
    if len(payload) != 4 + length:
        raise ProtocolError("FETCH payload length does not match its text")
    return payload[4:].decode("utf-8")


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


def encode_document(media_type: str, data: bytes) -> bytes:
    """Payload de ``OP_DOCUMENT``: media type + bytes del documento.

    Es la forma completa (header + datos) que usan tests y clientes. El servidor
    envía el header con :func:`encode_document_header` y luego los trozos del
    archivo, sin cargarlo entero en memoria.
    """
    return encode_document_header(media_type, len(data)) + data


def encode_document_header(media_type: str, size: int) -> bytes:
    """Header del payload ``OP_DOCUMENT``: ``mime + tamaño total de datos``."""
    mime = media_type.encode("utf-8")
    if not mime or len(mime) > 0xFFFF:
        raise ProtocolError("document media type is empty or too long")
    if size < 0 or size > 0xFFFFFFFF:
        raise ProtocolError("document size does not fit the u32 data length")
    return struct.pack(">H", len(mime)) + mime + struct.pack(">I", size)


def decode_document(payload: bytes) -> tuple[str, bytes]:
    if len(payload) < 6:
        raise ProtocolError("DOCUMENT payload shorter than its header")
    (mime_length,) = struct.unpack_from(">H", payload, 0)
    offset = 2 + mime_length
    if len(payload) < offset + 4:
        raise ProtocolError("DOCUMENT payload truncated before its data length")
    media_type = payload[2:offset].decode("utf-8")
    (data_length,) = struct.unpack_from(">I", payload, offset)
    offset += 4
    if len(payload) != offset + data_length:
        raise ProtocolError("DOCUMENT payload length does not match its data")
    return media_type, payload[offset:]


def encode_resultset(result: ResultSet, *, version: int = VERSION) -> bytes:
    """Serialize a ResultSet using the requested wire version."""
    if version == VERSION_V1:
        return _encode_resultset_v1(result)
    if version == VERSION:
        return _encode_resultset_v2(result)
    raise ProtocolError(f"unsupported protocol version {version}")


def decode_resultset(payload: bytes, *, version: int = VERSION) -> ResultSet:
    if version == VERSION_V1:
        return _decode_resultset_v1(payload)
    if version == VERSION:
        return _decode_resultset_v2(payload)
    raise ProtocolError(f"unsupported protocol version {version}")


def _encode_resultset_v1(result: ResultSet) -> bytes:
    """Legacy payload: numeric column type codes."""
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


def _decode_resultset_v1(payload: bytes) -> ResultSet:
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


def _encode_resultset_v2(result: ResultSet) -> bytes:
    """Schema-driven payload: type name, encoding and size per column."""
    out = bytearray()
    out += struct.pack(">H", len(result.columns))
    for column in result.columns:
        name = column.name.encode("utf-8")
        type_name = column.type_name.value.encode("utf-8")
        encoding, size = _wire_encoding(column.type_name)
        out += struct.pack(">H", len(name)) + name
        out += struct.pack(">H", len(type_name)) + type_name
        out += struct.pack(">BHH", encoding, size, column.length or 0)
    out += struct.pack(">I", len(result.rows))
    for row in result.rows:
        for value in row:
            _encode_value(out, value)
    return bytes(out)


def _decode_resultset_v2(payload: bytes) -> ResultSet:
    (count,) = struct.unpack_from(">H", payload, 0)
    offset = 2
    columns: list[ColumnDef] = []
    for _ in range(count):
        (name_len,) = struct.unpack_from(">H", payload, offset)
        offset += 2
        name = payload[offset : offset + name_len].decode("utf-8")
        offset += name_len
        (type_len,) = struct.unpack_from(">H", payload, offset)
        offset += 2
        type_name = payload[offset : offset + type_len].decode("utf-8")
        offset += type_len
        _encoding, _size, length = struct.unpack_from(">BHH", payload, offset)
        offset += 5
        columns.append(ColumnDef(name, ColumnType(type_name), length or None))
    (row_count,) = struct.unpack_from(">I", payload, offset)
    offset += 4
    rows: list[tuple[object, ...]] = []
    for _ in range(row_count):
        row: list[object] = []
        for _column in columns:
            value, offset = _decode_value(payload, offset)
            row.append(value)
        rows.append(tuple(row))
    return ResultSet(columns=tuple(columns), rows=tuple(rows))


def _wire_encoding(type_name: ColumnType) -> tuple[int, int]:
    """Return ``(encoding, size)`` for the schema-driven result header."""
    sizes = {
        ColumnType.INT: 4,
        ColumnType.FLOAT: 8,
        ColumnType.BOOL: 1,
        ColumnType.POINT: 16,
        ColumnType.VARCHAR: 0,
        ColumnType.TEXT: 0,
    }
    return 0, sizes[type_name]


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
    elif _is_point(value):
        # Dos doubles en el mismo orden que Point(x, y): x es longitud, y latitud.
        out += struct.pack(">Bdd", TAG_POINT, float(value[0]), float(value[1]))
    elif isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        out += struct.pack(">BI", TAG_BYTES, len(data)) + data
    else:
        raise ProtocolError(f"cannot encode value {value!r}")


def _is_point(value: object) -> bool:
    """Un POINT llega del motor como la tupla (x, y) que guarda el registro."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(
            isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
            for coordinate in value
        )
    )


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
    if tag == TAG_POINT:
        x, y = struct.unpack_from(">dd", payload, offset)
        return (x, y), offset + 16
    if tag == TAG_BYTES:
        (length,) = struct.unpack_from(">I", payload, offset)
        offset += 4
        return payload[offset : offset + length], offset + length
    raise ProtocolError(f"unknown value tag {tag}")


def _type_code(type_name: ColumnType) -> int:
    return {
        ColumnType.INT: TYPE_INT,
        ColumnType.FLOAT: TYPE_FLOAT,
        ColumnType.VARCHAR: TYPE_VARCHAR,
        ColumnType.TEXT: TYPE_TEXT,
        ColumnType.BOOL: TYPE_BOOL,
        ColumnType.POINT: TYPE_POINT,
    }[type_name]


def _column_type(code: int) -> ColumnType:
    return {
        TYPE_INT: ColumnType.INT,
        TYPE_FLOAT: ColumnType.FLOAT,
        TYPE_VARCHAR: ColumnType.VARCHAR,
        TYPE_TEXT: ColumnType.TEXT,
        TYPE_BOOL: ColumnType.BOOL,
        TYPE_POINT: ColumnType.POINT,
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
