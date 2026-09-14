"""Tests for the SQL lexer."""

import pytest

from engine.query.errors import QueryParseError
from engine.query.lexer import tokenize
from engine.query.tokens import TokenKind


def token_values(sql: str) -> list[tuple[TokenKind, str]]:
    return [(tok.kind, tok.value) for tok in tokenize(sql) if tok.kind is not TokenKind.EOF]


def test_select_star_where_tokens() -> None:
    toks = token_values("SELECT * FROM papers WHERE anio = 2020")
    assert toks == [
        (TokenKind.KEYWORD, "SELECT"),
        (TokenKind.STAR, "*"),
        (TokenKind.KEYWORD, "FROM"),
        (TokenKind.IDENTIFIER, "papers"),
        (TokenKind.KEYWORD, "WHERE"),
        (TokenKind.IDENTIFIER, "anio"),
        (TokenKind.OPERATOR, "="),
        (TokenKind.NUMBER, "2020"),
    ]


def test_keywords_are_normalized_uppercase() -> None:
    toks = token_values("select distinct id from papers order by anio desc")
    assert toks[0] == (TokenKind.KEYWORD, "SELECT")
    assert toks[1] == (TokenKind.KEYWORD, "DISTINCT")
    assert toks[3] == (TokenKind.KEYWORD, "FROM")
    assert toks[5] == (TokenKind.KEYWORD, "ORDER")
    assert toks[6] == (TokenKind.KEYWORD, "BY")
    assert toks[8] == (TokenKind.KEYWORD, "DESC")


def test_comparison_operators() -> None:
    toks = token_values("a <> b AND a <= 1 OR a >= 2 AND a != 3")
    ops = [v for k, v in toks if k is TokenKind.OPERATOR]
    assert ops == ["<>", "<=", ">=", "!="]


def test_string_literal_single_quotes() -> None:
    toks = token_values("titulo = 'RAG sobre papers'")
    assert toks[-1] == (TokenKind.STRING, "RAG sobre papers")


def test_parentheses_and_commas() -> None:
    toks = token_values("(id, titulo)")
    assert [(k, v) for k, v in toks] == [
        (TokenKind.LPAREN, "("),
        (TokenKind.IDENTIFIER, "id"),
        (TokenKind.COMMA, ","),
        (TokenKind.IDENTIFIER, "titulo"),
        (TokenKind.RPAREN, ")"),
    ]


def test_unterminated_string_raises() -> None:
    with pytest.raises(QueryParseError):
        tokenize("titulo = 'incompleto")


def test_tokens_carry_positions() -> None:
    tokens = [t for t in tokenize("SELECT id") if t.kind is not TokenKind.EOF]
    assert [_t.position for _t in tokens] == [0, 7]
