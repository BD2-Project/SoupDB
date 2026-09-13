"""Recursive descent SQL parser producing an AST.

The parser consumes the token stream from :mod:`engine.query.lexer` and builds
:mod:`engine.query.ast` nodes. Entry point is :func:`parse`.

Expression precedence (low to high):

``OR`` < ``AND`` < ``NOT`` < comparison (``= <> != < <= > >=``,
``BETWEEN``, ``IN``, ``LIKE``) < primary.
"""

from engine.query.ast import (
    BetweenExpr,
    ColumnRef,
    CompareExpr,
    Expr,
    InExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    SelectColumn,
    SelectStatement,
    Statement,
)
from engine.query.errors import QueryParseError
from engine.query.lexer import tokenize
from engine.query.tokens import Token, TokenKind

_COMPARISON_OPS = {"=", "<>", "!=", "<", "<=", ">", ">="}


def _number_value(raw: str) -> int | float:
    """Convert a NUMBER token into an int when possible, else a float."""
    if "." in raw:
        return float(raw)
    return int(raw)


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

    def _match_comma(self) -> bool:
        return self._match_kind(TokenKind.COMMA)

    def _match_kind(self, kind: TokenKind) -> bool:
        if self._check_kind(kind):
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
        distinct = self._match_keyword("DISTINCT")
        columns = self._parse_projection()
        self._expect_keyword("FROM")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        where = None
        if self._match_keyword("WHERE"):
            where = self._parse_boolean_expression()
        self._error_if_not_eof()
        return SelectStatement(columns=columns, table=table, where=where, distinct=distinct)

    def _parse_projection(self) -> tuple[SelectColumn, ...]:
        if self._match_kind(TokenKind.STAR):
            if self._check_kind(TokenKind.COMMA):
                token = self._peek()
                raise QueryParseError(f"star cannot mix with columns at position {token.position}")
            return ()
        columns = [self._parse_select_column()]
        while self._match_comma():
            columns.append(self._parse_select_column())
        return tuple(columns)

    def _parse_select_column(self) -> SelectColumn:
        expr = self._parse_expression()
        alias = None
        if self._match_keyword("AS"):
            alias = self._expect_kind(TokenKind.IDENTIFIER).value
        elif self._check_kind(TokenKind.IDENTIFIER):
            alias = self._advance().value
        return SelectColumn(expr=expr, alias=alias)

    def _parse_expression(self) -> Expr:
        """Parse any expression used in projections and clause keys.

        Comparisons are optional here: a bare column reference is a valid
        projection term.
        """
        return self._parse_or()

    def _parse_boolean_expression(self) -> Expr:
        """Parse a WHERE predicate, which must be a boolean-capable node."""
        expr = self._parse_expression()
        if isinstance(expr, (ColumnRef, Literal)):
            token = self._peek()
            raise QueryParseError(
                f"expected comparison at position {token.position}, got {token.value!r}"
            )
        return expr

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._match_keyword("OR"):
            right = self._parse_and()
            left = LogicalExpr(left, "OR", right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self._match_keyword("AND"):
            right = self._parse_not()
            left = LogicalExpr(left, "AND", right)
        return left

    def _parse_not(self) -> Expr:
        if self._match_keyword("NOT"):
            return NotExpr(self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self) -> Expr:
        value = self._parse_primary()
        token = self._peek()

        if token.kind is TokenKind.OPERATOR and token.value in _COMPARISON_OPS:
            self._advance()
            right = self._parse_primary()
            return CompareExpr(value, token.value, right)

        if self._match_keyword("BETWEEN"):
            lo = self._parse_primary()
            self._expect_keyword("AND")
            hi = self._parse_primary()
            return BetweenExpr(value, lo, hi)

        if self._match_keyword("IN"):
            self._expect_kind(TokenKind.LPAREN)
            items = [self._parse_primary()]
            while self._match_comma():
                items.append(self._parse_primary())
            self._expect_kind(TokenKind.RPAREN)
            return InExpr(value, tuple(items))

        if self._match_keyword("LIKE"):
            pattern = self._parse_primary()
            return LikeExpr(value, pattern)

        return value

    def _parse_primary(self) -> Expr:
        token = self._peek()

        if self._match_kind(TokenKind.NUMBER):
            return Literal(_number_value(token.value))

        if self._match_kind(TokenKind.STRING):
            return Literal(token.value)

        if self._match_kind(TokenKind.IDENTIFIER):
            return ColumnRef(token.value)

        if self._match_kind(TokenKind.LPAREN):
            expr = self._parse_or()
            self._expect_kind(TokenKind.RPAREN)
            return expr

        raise QueryParseError(
            f"expected expression at position {token.position}, got {token.value!r}"
        )


def parse(sql: str) -> Statement:
    """Tokenize and parse ``sql`` into its AST statement."""
    parser = _Parser(tokenize(sql))
    statement = parser.parse_statement()
    parser._error_if_not_eof()
    return statement
