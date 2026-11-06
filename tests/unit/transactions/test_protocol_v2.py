"""Wire protocol v2: schema-driven results and prepared statements."""

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query import execute_sql
from engine.query.params import bind_parameters
from engine.query.parser import parse
from engine.query.resultset import ResultSet
from engine.transactions import protocol as proto
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.session import TransactionalSession


def test_v2_resultset_carries_type_names_and_roundtrips() -> None:
    result = ResultSet(
        columns=(
            ColumnDef("id", ColumnType.INT),
            ColumnDef("titulo", ColumnType.TEXT),
            ColumnDef("pos", ColumnType.POINT),
        ),
        rows=((7, "hola", (1.5, 2.5)),),
    )
    payload = proto.encode_resultset(result, version=proto.VERSION)
    decoded = proto.decode_resultset(payload, version=proto.VERSION)
    assert decoded == result


def test_v1_resultset_still_decodes() -> None:
    result = ResultSet(columns=(ColumnDef("id", ColumnType.INT),), rows=((1,),))
    payload = proto.encode_resultset(result, version=proto.VERSION_V1)
    assert proto.decode_resultset(payload, version=proto.VERSION_V1) == result


def test_prepared_payloads_roundtrip() -> None:
    name, sql = proto.decode_prepare(proto.encode_prepare("p", "SELECT $1"))
    assert (name, sql) == ("p", "SELECT $1")
    assert proto.decode_deallocate(proto.encode_deallocate("p")) == "p"
    assert proto.decode_execute(proto.encode_execute("p", (1, "x", 2.5, True))) == (
        "p",
        (1, "x", 2.5, True),
    )


def test_bind_parameters_requires_every_position() -> None:
    statement = parse("SELECT id FROM docs WHERE id = $1 AND texto = $2")
    with pytest.raises(QueryExecutionError, match="at least 2"):
        bind_parameters(statement, (1,))
    bound = bind_parameters(statement, (1, "x"))
    assert "docs" in str(bound)


def test_prepared_statement_parses_once_and_reuses_ast(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "db")
    try:
        execute_sql("CREATE TABLE docs (id INT, texto TEXT)", catalog)
        execute_sql("INSERT INTO docs VALUES (1, 'alpha'), (2, 'beta')", catalog)
        session = TransactionalSession(catalog)
        handler = ConnectionHandler.__new__(ConnectionHandler)
        prepared = {}

        import engine.transactions.connection_handler as module

        calls = []
        original = module.parse

        def counting_parse(sql):
            calls.append(sql)
            return original(sql)

        monkeypatch.setattr(module, "parse", counting_parse)
        handler._prepare(
            proto.encode_prepare("p", "SELECT texto FROM docs WHERE id = $1"),
            prepared,
            proto.VERSION,
        )
        assert len(calls) == 1
        for ident, expected in ((1, b"alpha"), (2, b"beta")):
            frame = handler._execute_prepared(
                proto.encode_execute("p", (ident,)), prepared, session, proto.VERSION
            )
            opcode, payload = proto.decode_frame(frame)
            assert opcode == proto.OP_RESULT
            result = proto.decode_resultset(payload, version=proto.VERSION)
            assert result.rows == ((expected.decode(),),)
        assert len(calls) == 1
    finally:
        catalog.close()
