"""``EXPLAIN (FORMAT JSON)`` devuelve el árbol serializado.

El panel de plan del frontend necesita el árbol estructurado: no puede
reconstruirlo a partir de las líneas indentadas del dibujo de texto.
"""

import json
from pathlib import Path

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryParseError
from engine.query import execute_sql
from engine.query.parser import parse

CONTRATO = {"op", "detail", "rows", "elapsed_ms", "disk_reads", "disk_writes", "children"}


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    catalogo = Catalog(tmp_path)
    execute_sql(
        "CREATE TABLE alumnos (id INT PRIMARY KEY, nombre VARCHAR(100), nota INT)", catalogo
    )
    for identifier, nombre, nota in ((1, "Pérez", 16), (2, "García", 12), (3, "Chen", 18)):
        execute_sql(f"INSERT INTO alumnos VALUES ({identifier}, '{nombre}', {nota})", catalogo)
    yield catalogo
    catalogo.close()


def _plan(catalogo: Catalog, sql: str) -> dict:
    resultado = execute_sql(sql, catalogo)
    assert len(resultado.rows) == 1, "el árbol entero va en una sola fila"
    return json.loads(resultado.rows[0][0])


def _nodos(node: dict) -> list[dict]:
    return [node, *(item for child in node["children"] for item in _nodos(child))]


def test_parser_records_the_format() -> None:
    assert parse("EXPLAIN (FORMAT JSON) SELECT * FROM t").json_format is True
    assert parse("EXPLAIN ANALYZE (FORMAT JSON) SELECT * FROM t").json_format is True
    assert parse("EXPLAIN SELECT * FROM t").json_format is False


def test_an_unknown_format_is_rejected() -> None:
    with pytest.raises(QueryParseError, match="only JSON is supported"):
        parse("EXPLAIN (FORMAT TEXT) SELECT * FROM t")


def test_every_node_follows_the_contract(catalog: Catalog) -> None:
    plan = _plan(catalog, "EXPLAIN (FORMAT JSON) SELECT * FROM alumnos WHERE nota >= 14")
    for node in _nodos(plan):
        assert set(node) == CONTRATO
        assert isinstance(node["op"], str) and node["op"]
        assert isinstance(node["children"], list)


def test_analyze_reports_real_rows(catalog: Catalog) -> None:
    plan = _plan(
        catalog,
        "EXPLAIN ANALYZE (FORMAT JSON) SELECT * FROM alumnos WHERE nota >= 14 ORDER BY id",
    )
    # Sin ANALYZE el plan no se ejecuta y todas las métricas quedan en cero.
    assert plan["rows"] == 2
    scan = next(node for node in _nodos(plan) if node["op"] == "TableScan")
    assert scan["rows"] == 3


def test_without_analyze_the_metrics_are_zero(catalog: Catalog) -> None:
    plan = _plan(catalog, "EXPLAIN (FORMAT JSON) SELECT * FROM alumnos WHERE nota >= 14")
    assert all(node["rows"] == 0 for node in _nodos(plan))


def test_detail_is_readable(catalog: Catalog) -> None:
    plan = _plan(catalog, "EXPLAIN (FORMAT JSON) SELECT * FROM alumnos WHERE nota >= 14")
    filtro = next(node for node in _nodos(plan) if node["op"] == "Filter")
    assert filtro["detail"]["predicate"] == "nota >= 14"


def test_the_text_rendering_still_works(catalog: Catalog) -> None:
    # El formato de texto es el que se usa desde una consola; no debe cambiar.
    resultado = execute_sql("EXPLAIN SELECT * FROM alumnos WHERE nota >= 14", catalog)
    assert len(resultado.rows) > 1
    assert resultado.rows[0][0].lstrip().startswith("->")
