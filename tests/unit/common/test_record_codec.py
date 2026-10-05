"""Tests for the binary row codec.

Layout (one flag byte per column): ``0x00`` marks NULL and then no column
bytes follow. Otherwise ``0x01`` (INT/FLOAT/VARCHAR/TEXT) or ``0x01``/``0x02``
(BOOL: True/False) precedes the column payload.
"""

import struct

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType

SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("score", ColumnType.FLOAT),
    ColumnDef("titulo", ColumnType.VARCHAR, length=32),
    ColumnDef("cuerpo", ColumnType.TEXT),
    ColumnDef("activo", ColumnType.BOOL),
)
ROW = (7, 0.9, "RAG", "resumen del paper", True)


def test_encode_decode_roundtrip() -> None:
    data = encode_row(ROW, SCHEMA)
    assert isinstance(data, bytes)
    assert decode_row(data, SCHEMA) == ROW


def test_encode_empty_schema() -> None:
    data = encode_row((), ())
    assert data == b""
    assert decode_row(data, ()) == ()


def test_encode_int_little_endian() -> None:
    data = encode_row((1,), (ColumnDef("id", ColumnType.INT),))
    assert data == b"\x01" + (1).to_bytes(4, "little")


def test_decode_int() -> None:
    data = b"\x01" + (2020).to_bytes(4, "little")
    assert decode_row(data, (ColumnDef("id", ColumnType.INT),)) == (2020,)


def test_encode_decode_negative_int_roundtrip() -> None:
    data = encode_row((-5,), (ColumnDef("id", ColumnType.INT),))
    assert data == b"\x01" + (-5).to_bytes(4, "little", signed=True)
    assert decode_row(data, (ColumnDef("id", ColumnType.INT),)) == (-5,)


def test_decode_int_signed() -> None:
    column = ColumnDef("id", ColumnType.INT)
    assert decode_row(b"\x01" + (0xFF).to_bytes(4, "little"), (column,)) == (255,)
    assert decode_row(b"\x01" + (-1).to_bytes(4, "little", signed=True), (column,)) == (-1,)


def test_encode_int_32bit_bounds() -> None:
    column = ColumnDef("id", ColumnType.INT)
    lo = -(2**31)
    hi = 2**31 - 1
    assert decode_row(encode_row((lo,), (column,)), (column,)) == (lo,)
    assert decode_row(encode_row((hi,), (column,)), (column,)) == (hi,)


def test_encode_int_out_of_range_raises() -> None:
    column = ColumnDef("id", ColumnType.INT)
    with pytest.raises(QueryExecutionError, match="32-bit"):
        encode_row((2**31,), (column,))
    with pytest.raises(QueryExecutionError, match="32-bit"):
        encode_row((-(2**31) - 1,), (column,))


def test_encode_float() -> None:
    data = encode_row((3.5,), (ColumnDef("score", ColumnType.FLOAT),))
    assert data == b"\x01" + struct.pack("<d", 3.5)


def test_decode_float() -> None:
    data = b"\x01" + struct.pack("<d", -1.25)
    assert decode_row(data, (ColumnDef("score", ColumnType.FLOAT),)) == (-1.25,)


def test_encode_bool() -> None:
    assert encode_row((True,), (ColumnDef("ok", ColumnType.BOOL),)) == b"\x01"
    assert encode_row((False,), (ColumnDef("ok", ColumnType.BOOL),)) == b"\x02"


def test_decode_bool() -> None:
    schema = (ColumnDef("ok", ColumnType.BOOL),)
    assert decode_row(b"\x01", schema) == (True,)
    assert decode_row(b"\x02", schema) == (False,)


def test_encode_point() -> None:
    schema = (ColumnDef("ubicacion", ColumnType.POINT),)
    data = encode_row(((3.0, -1.5),), schema)
    assert data == b"\x01" + struct.pack("<dd", 3.0, -1.5)


def test_decode_point() -> None:
    schema = (ColumnDef("ubicacion", ColumnType.POINT),)
    data = b"\x01" + struct.pack("<dd", 1.5, 2.25)
    assert decode_row(data, schema) == ((1.5, 2.25),)


def test_encode_point_coerces_int_coordinates() -> None:
    schema = (ColumnDef("ubicacion", ColumnType.POINT),)
    data = encode_row(((3, -1),), schema)
    assert data == b"\x01" + struct.pack("<dd", 3.0, -1.0)
    assert decode_row(data, schema) == ((3.0, -1.0),)


def test_encode_wrong_point_type_raises() -> None:
    schema = (ColumnDef("ubicacion", ColumnType.POINT),)
    for value in [1, "ab", (1,), (1, 2, 3), (1, "a"), (None, None), (True, False)]:
        with pytest.raises(QueryExecutionError):
            encode_row((value,), schema)


def test_decode_invalid_point_flag_raises() -> None:
    schema = (ColumnDef("ubicacion", ColumnType.POINT),)
    with pytest.raises(QueryExecutionError, match="POINT"):
        decode_row(b"\x02", schema)


def test_decode_truncated_point_raises() -> None:
    schema = (ColumnDef("ubicacion", ColumnType.POINT),)
    data = b"\x01" + struct.pack("<d", 3.0) + b"\x00\x01"
    with pytest.raises(QueryExecutionError):
        decode_row(data, schema)


def test_encode_varchar_with_length_prefix() -> None:
    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=32),)
    data = encode_row(("RAG",), schema)
    assert data[:1] == b"\x01"
    (length,) = struct.unpack("<I", data[1:5])
    assert length == 3
    assert data[5:] == b"RAG"


def test_encode_varchar_utf8_length() -> None:
    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=32),)
    data = encode_row(("ñandú",), schema)
    assert decode_row(data, schema) == ("ñandú",)


def test_encode_varchar_over_length_raises() -> None:
    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=5),)
    with pytest.raises(QueryExecutionError):
        encode_row(("demasiado largo",), schema)


def test_decode_text() -> None:
    schema = (ColumnDef("cuerpo", ColumnType.TEXT),)
    data = encode_row(("cuerpo de muchas palabras",), schema)
    assert decode_row(data, schema) == ("cuerpo de muchas palabras",)


def test_encode_wrong_int_type_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    with pytest.raises(QueryExecutionError):
        encode_row(("1",), schema)


def test_encode_bool_as_int_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    with pytest.raises(QueryExecutionError):
        encode_row((True,), schema)


def test_encode_int_as_float_raises() -> None:
    schema = (ColumnDef("score", ColumnType.FLOAT),)
    with pytest.raises(QueryExecutionError):
        encode_row((1,), schema)


def test_decode_truncated_data_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    with pytest.raises(QueryExecutionError):
        decode_row(b"\x01\x02", schema)


def test_decode_trailing_bytes_raises() -> None:
    schema = (ColumnDef("id", ColumnType.INT),)
    data = encode_row((1,), schema) + b"\x00"
    with pytest.raises(QueryExecutionError):
        decode_row(data, schema)


def test_encode_wrong_row_length_raises() -> None:
    with pytest.raises(QueryExecutionError):
        encode_row((1,), SCHEMA)


def test_decode_with_invalid_bool_byte_raises() -> None:
    schema = (ColumnDef("activo", ColumnType.BOOL),)
    with pytest.raises(QueryExecutionError, match="BOOL"):
        decode_row(b"\x03", schema)


def test_decode_invalid_utf8_text_raises() -> None:
    raw = b"\x01" + struct.pack("<I", 2) + b"\xff\xfe"
    with pytest.raises(QueryExecutionError, match="UTF-8"):
        decode_row(raw, (ColumnDef("txt", ColumnType.TEXT),))


def test_record_holds_encoded_bytes() -> None:
    data = encode_row(ROW, SCHEMA)
    record = Record(data=data)
    assert decode_row(record.data, SCHEMA) == ROW


NULL_SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("score", ColumnType.FLOAT),
    ColumnDef("activo", ColumnType.BOOL),
    ColumnDef("titulo", ColumnType.VARCHAR, length=32),
    ColumnDef("cuerpo", ColumnType.TEXT),
)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (ColumnDef("id", ColumnType.INT), None),
        (ColumnDef("score", ColumnType.FLOAT), None),
        (ColumnDef("activo", ColumnType.BOOL), None),
        (ColumnDef("titulo", ColumnType.VARCHAR, length=32), None),
        (ColumnDef("cuerpo", ColumnType.TEXT), None),
    ],
    ids=["INT", "FLOAT", "BOOL", "VARCHAR", "TEXT"],
)
def test_encode_null_uses_only_flag_byte(column: ColumnDef, value: object) -> None:
    assert encode_row((value,), (column,)) == b"\x00"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (ColumnDef("id", ColumnType.INT), (None,)),
        (ColumnDef("score", ColumnType.FLOAT), (None,)),
        (ColumnDef("activo", ColumnType.BOOL), (None,)),
        (ColumnDef("titulo", ColumnType.VARCHAR, length=32), (None,)),
        (ColumnDef("cuerpo", ColumnType.TEXT), (None,)),
    ],
    ids=["INT", "FLOAT", "BOOL", "VARCHAR", "TEXT"],
)
def test_decode_null_flag_without_column_bytes(
    column: ColumnDef, expected: tuple[object, ...]
) -> None:
    assert decode_row(b"\x00", (column,)) == expected


@pytest.mark.parametrize(
    "row",
    [
        (None, None, None, None, None),
        (1, None, True, None, None),
        (None, 2.5, False, "texto", "cuerpo"),
        (-3, 0.0, True, "", ""),
    ],
    ids=["all-null", "mixed-a", "mixed-b", "empty-text"],
)
def test_encode_decode_null_roundtrip(row: tuple[object, ...]) -> None:
    assert decode_row(encode_row(row, NULL_SCHEMA), NULL_SCHEMA) == row


def test_null_does_not_consume_following_column_bytes() -> None:
    row = (None, 42)
    schema = (ColumnDef("a", ColumnType.INT), ColumnDef("b", ColumnType.INT))
    data = encode_row(row, schema)
    assert data == b"\x00\x01" + (42).to_bytes(4, "little")
    assert decode_row(data, schema) == row


def test_null_varchar_omits_length_prefix() -> None:
    schema = (ColumnDef("titulo", ColumnType.VARCHAR, length=32),)
    assert encode_row((None,), schema) == b"\x00"
