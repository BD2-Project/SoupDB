"""Comparar INT con FLOAT no debe fallar.

Un radio se escribe `< 5000`, no `< 5000.0`, y `distance(...)` siempre devuelve
float: sin promoción numérica el ejemplo de consulta espacial del enunciado
termina en un error de tipos.
"""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.evaluator import evaluate
from engine.query.parser import parse

SCHEMA = (
    ColumnDef("entero", ColumnType.INT),
    ColumnDef("real", ColumnType.FLOAT),
    ColumnDef("bandera", ColumnType.BOOL),
)
ROW = (3, 2.5, True)


def _where(sql: str) -> object:
    statement = parse(f"SELECT * FROM t WHERE {sql}")
    return evaluate(statement.where, ROW, SCHEMA)


@pytest.mark.parametrize(
    "expression",
    [
        "real < 3",
        "3 > real",
        "real = 2.5",
        "entero < 3.5",
        "entero >= 3.0",
        "real BETWEEN 0 AND 3",
        "entero BETWEEN 2.5 AND 3.5",
        "entero IN (1, 2.0, 3)",
    ],
)
def test_int_and_float_compare(expression: str) -> None:
    assert _where(expression) is True


@pytest.mark.parametrize("expression", ["bandera = 1", "entero = TRUE", "real > TRUE"])
def test_bool_is_not_a_number(expression: str) -> None:
    # BOOL hereda de int en Python, pero comparar una bandera con un número no
    # tiene sentido en SQL y debe seguir fallando.
    with pytest.raises(QueryExecutionError, match="cannot mix"):
        _where(expression)
