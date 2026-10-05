"""`distancia(...)` es el nombre que usa el enunciado para la función de distancia.

Se acepta como alias de `distance(...)`, pero solo cuando abre una llamada: de lo
contrario `distancia` dejaría de servir como nombre de columna, que es un nombre
muy probable en una base de datos escrita en español.
"""

import pytest

from engine.query.lexer import TokenKind, tokenize


def _kinds(sql: str) -> list[tuple[TokenKind, str]]:
    return [(token.kind, token.value) for token in tokenize(sql)]


@pytest.mark.parametrize("sql", ["distancia(a, b)", "DISTANCIA(a, b)", "distancia (a, b)"])
def test_alias_becomes_the_distance_keyword(sql: str) -> None:
    assert (TokenKind.KEYWORD, "DISTANCE") in _kinds(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT distancia FROM t",
        "SELECT id AS distancia FROM t",
        "SELECT * FROM t WHERE distancia > 8",
    ],
)
def test_alias_is_still_a_valid_identifier(sql: str) -> None:
    tokens = _kinds(sql)
    assert (TokenKind.IDENTIFIER, "distancia") in tokens
    assert (TokenKind.KEYWORD, "DISTANCE") not in tokens
