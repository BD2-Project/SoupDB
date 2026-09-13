"""Recursive descent SQL parser producing an AST.

The parser consumes the token stream from :mod:`engine.query.lexer` and builds
:mod:`engine.query.ast` nodes. Entry point is :func:`parse`.
"""

from engine.query.ast import ColumnRef, SelectColumn, SelectStatement, Statement
from engine.query.errors import QueryParseError
from engine.query.lexer import tokenize
from engine.query.tokens import Token, TokenKind


class _Parser:
    """Internal parser over the token stream."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def _check_keyword(self, keyword: str) -> bool:
        token = self._peek()
        return token.kind is TokenKind.KEYWORD and token.value == keyword

    def _check_kind(self, kind: TokenKind) -> bool:
        return self._peek().kind is kind

    def _match_keyword(self, keyword: str) -> bool:
        if self._check_keyword(keyword):
            self._advance()
            return True
        return False

    def _expect_keyword(self, keyword: str) -> None:
        if not self._match_keyword(keyword):
            token = self._peek()
            raise QueryParseError(
                f"expected {keyword} at position {token.position}, got {token.value!r}"
            )

    def _expect_kind(self, kind: TokenKind) -> Token:
        token = self._peek()
        if token.kind is not kind:
            raise QueryParseError(
                f"expected {kind.value} at position {token.position}, got {token.value!r}"
            )
        return self._advance()

    def _error_if_not_eof(self) -> None:
        token = self._peek()
        if token.kind is not TokenKind.EOF:
            raise QueryParseError(f"unexpected token {token.value!r} at position {token.position}")

    def parse_statement(self) -> Statement:
        if self._match_keyword("SELECT"):
            return self._parse_select()
        raise QueryParseError(f"unsupported statement at position {self._peek().position}")

    def _parse_select(self) -> SelectStatement:
        columns = self._parse_projection()
        self._expect_keyword("FROM")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        self._error_if_not_eof()
        return SelectStatement(columns=columns, table=table)

    def _parse_projection(self) -> tuple[SelectColumn, ...]:
        if self._match_kind_star():
            return ()
        columns = [self._parse_select_column()]
        while self._match_comma():
            columns.append(self._parse_select_column())
        return tuple(columns)

    def _parse_select_column(self) -> SelectColumn:
        expr = self._parse_column_ref()
        return SelectColumn(expr=expr)

    def _parse_column_ref(self) -> ColumnRef:
        token = self._expect_kind(TokenKind.IDENTIFIER)
        return ColumnRef(token.value)

    def _match_kind_star(self) -> bool:
        if self._check_kind(TokenKind.STAR):
            self._advance()
            return True
        return False

    def _match_comma(self) -> bool:
        if self._check_kind(TokenKind.COMMA):
            self._advance()
            return True
        return False


def parse(sql: str) -> Statement:
    """Tokenize and parse ``sql`` into its AST statement."""
    parser = _Parser(tokenize(sql))
    statement = parser.parse_statement()
    parser._error_if_not_eof()
    return statement
