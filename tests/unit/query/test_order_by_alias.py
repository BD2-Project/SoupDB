"""ORDER BY resolution of projection aliases.

``Sort`` is planned below ``Project`` so a key can name a column that is not
projected, which means the projection aliases are not in the schema the keys
are validated against. The planner resolves them explicitly: an ORDER BY key
that names an alias is rewritten to the aliased expression before validation,
and an alias shadows a base column with the same name.
"""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import BinaryExpr, ColumnRef, Literal
from engine.query.executor import execute
from engine.query.format_expr import format_expr
from engine.query.parser import parse
from engine.query.planner import Plan, plan
from tests.fakes.fake_catalog import FakeCatalog

ITEMS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("precio", ColumnType.INT),
    ColumnDef("categoria", ColumnType.TEXT),
)
ITEMS_ROWS = (
    (1, 10, "a"),
    (2, 30, "b"),
    (3, 20, "a"),
    (4, 20, "b"),
    (5, 5, "b"),
)

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)
LUGARES_ROWS = (("A", (0.0, 0.0)), ("B", (3.0, 4.0)), ("C", (6.0, 8.0)))

DISTANCE_FROM_ORIGIN = "distance(ubicacion, POINT(0, 0))"


def make_items() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("items", ITEMS)
    for row in ITEMS_ROWS:
        catalog.insert("items", row)
    return catalog


def make_lugares() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("lugares", LUGARES)
    for row in LUGARES_ROWS:
        catalog.insert("lugares", row)
    return catalog


def plan_select(sql: str, catalog: FakeCatalog) -> Plan:
    return plan(parse(sql), catalog)


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def sort_keys(sql: str, catalog: FakeCatalog) -> list[str]:
    """Keys of the Sort node of a planned select, as EXPLAIN would print them."""
    sort = plan_select(sql, catalog).root.explain().children[0]
    assert sort.op == "Sort"
    return sort.detail["keys"]


def test_order_by_alias_of_spatial_expression() -> None:
    result = run(
        f"SELECT nombre, {DISTANCE_FROM_ORIGIN} AS d FROM lugares ORDER BY d",
        make_lugares(),
    )
    assert [column.name for column in result.columns] == ["nombre", "d"]
    assert result.rows == (("A", 0.0), ("B", 5.0), ("C", 10.0))


def test_order_by_alias_of_spatial_expression_desc() -> None:
    result = run(
        f"SELECT nombre, {DISTANCE_FROM_ORIGIN} AS d FROM lugares ORDER BY d DESC",
        make_lugares(),
    )
    assert result.rows == (("C", 10.0), ("B", 5.0), ("A", 0.0))


def test_order_by_alias_of_spatial_expression_with_limit() -> None:
    result = run(
        f"SELECT nombre, {DISTANCE_FROM_ORIGIN} AS d FROM lugares ORDER BY d LIMIT 2",
        make_lugares(),
    )
    assert result.rows == (("A", 0.0), ("B", 5.0))


def test_order_by_alias_of_arithmetic_expression_rewrites_the_key() -> None:
    catalog = make_items()
    result = plan_select("SELECT precio * 2 AS p FROM items ORDER BY p", catalog)
    tree = result.root.explain()
    assert tree.op == "Project"
    assert [column.name for column in result.output_schema] == ["p"]
    assert sort_keys("SELECT precio * 2 AS p FROM items ORDER BY p", catalog) == [
        format_expr(BinaryExpr(ColumnRef("precio"), "*", Literal(2)))
    ]


def test_order_by_alias_of_projected_column() -> None:
    result = run("SELECT precio AS p FROM items ORDER BY p DESC", make_items())
    assert result.rows == ((30,), (20,), (20,), (10,), (5,))


def test_order_by_alias_of_aggregate() -> None:
    result = run(
        "SELECT categoria, COUNT(*) AS n FROM items GROUP BY categoria ORDER BY n",
        make_items(),
    )
    assert result.rows == (("a", 2), ("b", 3))


def test_order_by_alias_of_aggregate_desc() -> None:
    result = run(
        "SELECT categoria, COUNT(*) AS n FROM items GROUP BY categoria ORDER BY n DESC",
        make_items(),
    )
    assert result.rows == (("b", 3), ("a", 2))


def test_order_by_alias_of_aggregate_key_is_the_aggregate_column() -> None:
    keys = sort_keys(
        "SELECT categoria, COUNT(*) AS n FROM items GROUP BY categoria ORDER BY n",
        make_items(),
    )
    assert keys == ["count_1"]


def test_order_by_aggregate_without_alias_still_resolves() -> None:
    catalog = make_items()
    sql = "SELECT categoria, COUNT(*) AS n FROM items GROUP BY categoria ORDER BY COUNT(*)"
    assert run(sql, catalog).rows == (("a", 2), ("b", 3))
    assert sort_keys(sql, catalog) == ["count_1"]


def test_order_by_alias_shadows_base_column() -> None:
    catalog = make_items()
    result = run("SELECT categoria AS id FROM items ORDER BY id", catalog)
    assert result.rows == (("a",), ("a",), ("b",), ("b",), ("b",))
    assert sort_keys("SELECT categoria AS id FROM items ORDER BY id", catalog) == ["categoria"]


def test_order_by_alias_shadows_base_column_desc() -> None:
    result = run("SELECT categoria AS id FROM items ORDER BY id DESC", make_items())
    assert result.rows == (("b",), ("b",), ("b",), ("a",), ("a",))


def test_order_by_non_projected_column_still_works() -> None:
    catalog = make_items()
    result = run("SELECT categoria FROM items ORDER BY precio", catalog)
    assert result.rows == (("b",), ("a",), ("a",), ("b",), ("b",))
    assert sort_keys("SELECT categoria FROM items ORDER BY precio", catalog) == ["precio"]


def test_order_by_two_aliases_with_mixed_direction() -> None:
    result = run(
        "SELECT precio AS p, categoria AS c FROM items ORDER BY c DESC, p",
        make_items(),
    )
    assert result.rows == ((5, "b"), (20, "b"), (30, "b"), (10, "a"), (20, "a"))


def test_order_by_alias_with_distinct() -> None:
    result = run("SELECT DISTINCT categoria AS c FROM items ORDER BY c DESC", make_items())
    assert result.rows == (("b",), ("a",))


def test_explain_order_by_alias_keeps_sort_below_project() -> None:
    result = run(
        f"EXPLAIN SELECT nombre, {DISTANCE_FROM_ORIGIN} AS d FROM lugares ORDER BY d",
        make_lugares(),
    )
    lines = [row[0] for row in result.rows]
    assert lines[0].startswith("-> Project")
    assert lines[1].startswith("  -> Sort")
    # El detalle se lee como SQL, no como el repr del AST.
    assert "distancia(" in lines[1]


def test_order_by_unknown_identifier_still_raises() -> None:
    with pytest.raises(QueryExecutionError, match="unknown column 'faltante'"):
        run("SELECT categoria AS c FROM items ORDER BY faltante", make_items())


def test_order_by_alias_is_case_sensitive() -> None:
    with pytest.raises(QueryExecutionError, match="unknown column 'C'"):
        run("SELECT categoria AS c FROM items ORDER BY C", make_items())


def test_order_by_alias_is_not_available_in_where() -> None:
    with pytest.raises(QueryExecutionError, match="unknown column 'c'"):
        run("SELECT categoria AS c FROM items WHERE c = 'a'", make_items())
