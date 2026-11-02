"""C21: wire protocol must preserve ranked FLOAT values and errors."""

from engine.common.schema import ColumnDef, ColumnType
from engine.query.resultset import ResultSet
from engine.transactions import protocol as proto


def test_ranked_float_roundtrip_keeps_sign_and_precision() -> None:
    source = ResultSet(
        columns=(
            ColumnDef("id", ColumnType.INT),
            ColumnDef("relevancia", ColumnType.FLOAT),
        ),
        rows=((7, 0.75), (8, -0.125), (9, 0.0)),
    )
    payload = proto.encode_resultset(source)
    result = proto.decode_resultset(payload)
    assert result.columns == source.columns
    assert result.rows == source.rows
    assert all(type(row[-1]) is float for row in result.rows)


def test_ranked_resultset_can_be_wrapped_in_wire_frame() -> None:
    source = ResultSet(
        columns=(ColumnDef("score", ColumnType.FLOAT),),
        rows=((0.95,), (-0.5,)),
    )
    packet = proto.encode_frame(proto.OP_RESULT, proto.encode_resultset(source))
    opcode, payload = proto.decode_frame(packet)
    assert opcode == proto.OP_RESULT
    assert proto.decode_resultset(payload).rows == ((0.95,), (-0.5,))


def test_ranked_execution_error_keeps_error_code() -> None:
    frame = proto.encode_frame(
        proto.OP_ERROR,
        proto.encode_error(proto.ERR_EXECUTION, "index metadata incompatible"),
    )
    opcode, payload = proto.decode_frame(frame)
    assert opcode == proto.OP_ERROR
    assert proto.decode_error(payload) == (
        proto.ERR_EXECUTION,
        "index metadata incompatible",
    )
