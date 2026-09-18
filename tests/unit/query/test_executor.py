"""Tests for the query executor across SELECT, INSERT, DELETE and CREATE."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("venue", ColumnType.TEXT),
)
ROWS = (
    (1, 2019, "VLDB"),
    (2, 2020, "VLDB"),
    (3, 2021, "SIGMOD"),
    (4, 2020, "SIGMOD"),
)


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def make_catalog(rows: tuple[tuple[object, ...], ...] = ROWS) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("papers", PAPERS)
    for row in rows:
        catalog.insert("papers", row)
    return catalog


def test_select_star_returns_rows() -> None:
    result = run("SELECT * FROM papers", make_catalog())
    assert [column.name for column in result.columns] == ["id", "anio", "venue"]
    assert result.rows == ROWS


def test_select_projection_with_alias() -> None:
    result = run("SELECT venue AS v FROM papers", make_catalog())
    assert [column.name for column in result.columns] == ["v"]
    assert result.rows == (("VLDB",), ("VLDB",), ("SIGMOD",), ("SIGMOD",))


def test_select_where_equality() -> None:
    result = run("SELECT * FROM papers WHERE anio = 2020", make_catalog())
    assert result.rows == ((2, 2020, "VLDB"), (4, 2020, "SIGMOD"))


def test_select_where_with_index() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio")
    result = run("SELECT * FROM papers WHERE anio = 2020", catalog)
    assert result.rows == ((2, 2020, "VLDB"), (4, 2020, "SIGMOD"))


def test_select_order_by() -> None:
    result = run("SELECT * FROM papers ORDER BY anio DESC", make_catalog())
    assert result.rows == (
        (3, 2021, "SIGMOD"),
        (2, 2020, "VLDB"),
        (4, 2020, "SIGMOD"),
        (1, 2019, "VLDB"),
    )


def test_select_distinct() -> None:
    result = run("SELECT DISTINCT anio FROM papers", make_catalog())
    assert sorted(row[0] for row in result.rows) == [2019, 2020, 2021]


def test_select_group_by() -> None:
    result = run(
        "SELECT venue, COUNT(*) FROM papers GROUP BY venue",
        make_catalog(),
    )
    assert result.rows == (("VLDB", 2), ("SIGMOD", 2))


def test_select_group_by_order_by_aggregate() -> None:
    loose = make_catalog()
    loose.insert("papers", (5, 2019, "VLDB"))
    result = run(
        "SELECT venue, COUNT(*) FROM papers GROUP BY venue ORDER BY COUNT(*)",
        loose,
    )
    assert result.rows == (("SIGMOD", 2), ("VLDB", 3))


def test_select_empty_table_unknown_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        run("SELECT * FROM papers WHERE missing = 1", make_catalog(rows=()))


def test_insert_returns_affected_and_persists() -> None:
    catalog = make_catalog()
    result = run("INSERT INTO papers VALUES (5, 2022, 'VLDB')", catalog)
    assert result.rows == ()
    assert result.affected == 1
    assert len(run("SELECT * FROM papers", catalog).rows) == 5


def test_insert_reordered_columns() -> None:
    catalog = make_catalog(rows=())
    result = run("INSERT INTO papers (venue, id, anio) VALUES ('ICDE', 9, 2020)", catalog)
    assert result.affected == 1
    assert run("SELECT * FROM papers", catalog).rows == ((9, 2020, "ICDE"),)


def test_insert_maintains_indexes() -> None:
    catalog = make_catalog(rows=())
    catalog.add_index("papers", "anio")
    result = run("INSERT INTO papers VALUES (5, 2022, 'VLDB')", catalog)
    assert result.affected == 1
    assert run("SELECT * FROM papers WHERE anio = 2022", catalog).rows == ((5, 2022, "VLDB"),)
    anio_index = catalog.indexes("papers")["anio"]
    assert len(anio_index.search(2022)) == 1


def test_insert_reordered_columns_maintains_indexes() -> None:
    catalog = make_catalog(rows=())
    catalog.add_index("papers", "anio")
    run("INSERT INTO papers (venue, id, anio) VALUES ('ICDE', 9, 2020)", catalog)
    assert run("SELECT * FROM papers WHERE anio = 2020", catalog).rows == ((9, 2020, "ICDE"),)


def test_delete_filtered() -> None:
    catalog = make_catalog()
    result = run("DELETE FROM papers WHERE anio = 2020", catalog)
    assert result.affected == 2
    assert len(run("SELECT * FROM papers", catalog).rows) == 2


def test_delete_empty_table_unknown_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        run("DELETE FROM papers WHERE missing = 1", make_catalog(rows=()))


def test_delete_all() -> None:
    catalog = make_catalog()
    result = run("DELETE FROM papers", catalog)
    assert result.affected == 4
    assert run("SELECT * FROM papers", catalog).rows == ()


def test_delete_maintains_index() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio")
    result = run("DELETE FROM papers WHERE anio = 2020", catalog)
    assert result.affected == 2
    assert run("SELECT * FROM papers WHERE anio = 2020", catalog).rows == ()
    assert run("SELECT * FROM papers WHERE anio = 2019", catalog).rows == ((1, 2019, "VLDB"),)
    anio_index = catalog.indexes("papers")["anio"]
    assert anio_index.search(2020) == []
    assert len(anio_index.search(2019)) == 1


def test_delete_all_maintains_index() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio")
    result = run("DELETE FROM papers", catalog)
    assert result.affected == 4
    anio_index = catalog.indexes("papers")["anio"]
    assert anio_index.search(2020) == []
    assert anio_index.search(2019) == []


def test_create_table_persists_schema() -> None:
    catalog = FakeCatalog()
    result = run("CREATE TABLE fresh (a INT, b TEXT)", catalog)
    assert result.columns == ()
    assert result.affected == 0
    run("INSERT INTO fresh VALUES (1, 'x')", catalog)
    assert run("SELECT * FROM fresh", catalog).rows == ((1, "x"),)


def test_select_unknown_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        run("SELECT missing FROM papers", make_catalog())


def test_create_table_duplicate_raises() -> None:
    catalog = FakeCatalog()
    with pytest.raises(QueryExecutionError):
        run("CREATE TABLE t (a INT, a INT)", catalog)


def test_drop_table_removes_table() -> None:
    catalog = make_catalog()
    result = run("DROP TABLE papers", catalog)
    assert result.affected == 0
    with pytest.raises(QueryExecutionError):
        run("SELECT * FROM papers", catalog)


def test_drop_table_unknown_raises() -> None:
    with pytest.raises(QueryExecutionError):
        run("DROP TABLE nope", make_catalog())


def test_drop_index_removes_index() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio", index_name="idx_anio")
    run("DROP INDEX idx_anio", catalog)
    assert run("SELECT * FROM papers WHERE anio = 2020", catalog).rows != ()


def test_drop_index_unknown_raises() -> None:
    with pytest.raises(QueryExecutionError):
        run("DROP INDEX nope", make_catalog())
