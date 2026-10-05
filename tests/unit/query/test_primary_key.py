"""``PRIMARY KEY`` en la definición de columna.

No es una anotación decorativa: declarar la clave crea un índice B+ sobre esa
columna, que es lo que hace que una búsqueda por clave no recorra la tabla.
"""

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError, QueryParseError
from engine.query import execute_sql
from engine.query.parser import parse

CREATE = "CREATE TABLE alumnos (id INT PRIMARY KEY, nombre VARCHAR(100), nota INT)"


def test_parser_records_the_primary_key() -> None:
    statement = parse(CREATE)
    assert statement.primary_key == "id"
    assert [column.name for column in statement.columns] == ["id", "nombre", "nota"]


def test_without_primary_key_the_field_is_none() -> None:
    statement = parse("CREATE TABLE t (id INT, nombre TEXT)")
    assert statement.primary_key is None


def test_two_primary_keys_are_rejected() -> None:
    with pytest.raises(QueryParseError, match="more than one PRIMARY KEY"):
        parse("CREATE TABLE t (a INT PRIMARY KEY, b INT PRIMARY KEY)")


def test_primary_key_creates_an_index(tmp_path) -> None:
    catalog = Catalog(tmp_path)
    execute_sql(CREATE, catalog)

    indexes = execute_sql("SELECT * FROM SysIndexes", catalog).rows

    assert ("alumnos_pk", "alumnos", "id", "BTREE") in indexes
    catalog.close()


def test_primary_key_on_an_unknown_column_is_rejected() -> None:
    # No se puede escribir hoy con la gramática actual, pero el planner debe
    # rechazarlo igual: la validación no depende de cómo se construyó el AST.
    from engine.common.schema import ColumnDef, ColumnType
    from engine.query.ast import CreateTableStatement
    from engine.query.planner import plan

    statement = CreateTableStatement(
        table="t",
        columns=(ColumnDef("a", ColumnType.INT),),
        primary_key="inexistente",
    )
    with pytest.raises(QueryExecutionError, match="PRIMARY KEY"):
        plan(statement, None)


def test_the_evaluation_script_runs(tmp_path) -> None:
    """Las consultas con las que se evalúa el proyecto, de principio a fin."""
    catalog = Catalog(tmp_path)
    execute_sql(
        "CREATE TABLE alumnos (id INT PRIMARY KEY, nombre VARCHAR(100), carrera_id INT, nota INT)",
        catalog,
    )
    for identifier, nombre, carrera, nota in (
        (1, "Pérez, Juan", 1, 16),
        (2, "García, Ana", 1, 12),
        (3, "Chen, Li", 2, 18),
    ):
        execute_sql(
            f"INSERT INTO alumnos VALUES ({identifier}, '{nombre}', {carrera}, {nota})",
            catalog,
        )

    assert len(execute_sql("SELECT * FROM alumnos WHERE nombre = 'Pérez, Juan'", catalog).rows) == 1
    assert len(execute_sql("SELECT * FROM alumnos WHERE nota >= 14 ORDER BY id", catalog).rows) == 2
    assert execute_sql("SELECT * FROM alumnos WHERE id = 999", catalog).rows == ()
    assert execute_sql("EXPLAIN SELECT * FROM alumnos WHERE nota >= 14 ORDER BY id", catalog).rows
    assert execute_sql(
        "EXPLAIN ANALYZE SELECT * FROM alumnos WHERE nota >= 14 ORDER BY id", catalog
    ).rows
    catalog.close()
