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


def test_update_and_set_keywords() -> None:
    toks = token_values("UPDATE papers SET titulo = 'nuevo'")
    assert toks[0] == (TokenKind.KEYWORD, "UPDATE")
    assert toks[2] == (TokenKind.KEYWORD, "SET")


def test_join_inner_on_keywords() -> None:
    toks = token_values("SELECT * FROM a JOIN b INNER ON a = b")
    assert (TokenKind.KEYWORD, "JOIN") in toks
    assert (TokenKind.KEYWORD, "INNER") in toks
    assert (TokenKind.KEYWORD, "ON") in toks


def test_having_limit_offset_explain_keywords() -> None:
    toks = token_values("EXPLAIN SELECT id FROM t HAVING count > 1 LIMIT 10 OFFSET 5")
    assert toks[0] == (TokenKind.KEYWORD, "EXPLAIN")
    assert (TokenKind.KEYWORD, "HAVING") in toks
    assert (TokenKind.KEYWORD, "LIMIT") in toks
    assert (TokenKind.KEYWORD, "OFFSET") in toks


def test_is_and_null_keywords() -> None:
    toks = token_values("SELECT * FROM t WHERE autor IS NULL")
    assert (TokenKind.KEYWORD, "IS") in toks
    assert (TokenKind.NULL, "NULL") in toks


def test_join_keywords_normalized_uppercase() -> None:
    toks = token_values("explain select * from a inner join b on a = b where x is null")
    assert (TokenKind.KEYWORD, "EXPLAIN") in toks
    assert (TokenKind.KEYWORD, "INNER") in toks
    assert (TokenKind.KEYWORD, "JOIN") in toks
    assert (TokenKind.KEYWORD, "IS") in toks
    assert (TokenKind.NULL, "NULL") in toks


def test_new_keywords_are_not_identifiers() -> None:
    toks = token_values("update set join inner having limit offset is null explain")
    assert all(kind is not TokenKind.IDENTIFIER for kind, _ in toks)


def test_comparison_operators() -> None:
    toks = token_values("a <> b AND a <= 1 OR a >= 2 AND a != 3")
    ops = [v for k, v in toks if k is TokenKind.OPERATOR]
    assert ops == ["<>", "<=", ">=", "!="]


def test_decimal_number_token() -> None:
    toks = token_values("SELECT 1.25 FROM papers")
    assert (TokenKind.NUMBER, "1.25") in toks


def test_negative_number_tokens() -> None:
    toks = token_values("SELECT * FROM papers WHERE anio = -1 AND score >= -1.5")
    assert (TokenKind.NUMBER, "-1") in toks
    assert (TokenKind.NUMBER, "-1.5") in toks


def test_arithmetic_operators() -> None:
    toks = token_values("a + b - c * d / e % f")
    ops = [v for k, v in toks if k is TokenKind.OPERATOR]
    assert ops == ["+", "-", "/", "%"]
    assert (TokenKind.STAR, "*") in toks


def test_arithmetic_and_comparison_mix() -> None:
    toks = token_values("a + 1 >= b - 2 AND c <= 3")
    ops = [v for k, v in toks if k is TokenKind.OPERATOR]
    assert ops == ["+", ">=", "-", "<="]


def test_minus_operator_between_numbers() -> None:
    toks = token_values("5 - 1")
    assert (TokenKind.NUMBER, "5") in toks
    assert (TokenKind.OPERATOR, "-") in toks
    assert (TokenKind.NUMBER, "1") in toks


def test_trailing_minus_is_operator_not_error() -> None:
    toks = token_values("a = 5 -")
    assert toks == [
        (TokenKind.IDENTIFIER, "a"),
        (TokenKind.OPERATOR, "="),
        (TokenKind.NUMBER, "5"),
        (TokenKind.OPERATOR, "-"),
    ]


def test_string_literal_single_quotes() -> None:
    toks = token_values("titulo = 'RAG sobre papers'")
    assert toks[-1] == (TokenKind.STRING, "RAG sobre papers")


def test_string_literal_escaped_apostrophe() -> None:
    toks = token_values("autor = 'O''Reilly'")
    assert toks[-1] == (TokenKind.STRING, "O'Reilly")


def test_string_token_position_points_at_opening_quote() -> None:
    tokens = [t for t in tokenize("autor = 'x'") if t.kind is TokenKind.STRING]
    assert tokens[0].position == 8


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
