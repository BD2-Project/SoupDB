"""Recursive descent SQL parser producing an AST.

The parser consumes the token stream from :mod:`engine.query.lexer` and builds
:mod:`engine.query.ast` nodes. Entry point is :func:`parse`.

Expression precedence (low to high):

``OR`` < ``AND`` < ``NOT`` < comparison (``= <> != < <= > >=``,
``BETWEEN``, ``IN``, ``LIKE``) < primary.
"""

from engine.query.ast import (
    BetweenExpr,
    ColumnDef,
    ColumnRef,
    ColumnType,
    CompareExpr,
    CreateIndexStatement,
    CreateTableStatement,
    DeleteStatement,
    Expr,
    FunctionExpr,
    InExpr,
    InsertStatement,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    OrderByItem,
    SelectColumn,
    SelectStatement,
    Statement,
)
from engine.query.errors import QueryParseError
from engine.query.lexer import tokenize
from engine.query.tokens import Token, TokenKind

_COMPARISON_OPS = {"=", "<>", "!=", "<", "<=", ">", ">="}
_AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


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

    def _expect_ident_or_keyword(self) -> str:
        """Consume a bare word that can arrive as identifier or keyword."""
        token = self._peek()
        if token.kind in (TokenKind.IDENTIFIER, TokenKind.KEYWORD):
            return self._advance().value.upper()
        raise QueryParseError(f"expected a name at position {token.position}, got {token.value!r}")

    def _error_if_not_eof(self) -> None:
        self._match_kind(TokenKind.SEMICOLON)
        token = self._peek()
        if token.kind is not TokenKind.EOF:
            raise QueryParseError(f"unexpected token {token.value!r} at position {token.position}")

    def parse_statement(self) -> Statement:
        if self._match_keyword("SELECT"):
            return self._parse_select()
        if self._match_keyword("DELETE"):
            return self._parse_delete()
        if self._match_keyword("INSERT"):
            return self._parse_insert()
        if self._match_keyword("CREATE"):
            if self._match_keyword("INDEX"):
                return self._parse_create_index()
            return self._parse_create_table()
        raise QueryParseError(f"unsupported statement at position {self._peek().position}")

    def _parse_delete(self) -> DeleteStatement:
        self._expect_keyword("FROM")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        where = None
        if self._match_keyword("WHERE"):
            where = self._parse_boolean_expression()
        self._error_if_not_eof()
        return DeleteStatement(table=table, where=where)

    def _parse_insert(self) -> InsertStatement:
        self._expect_keyword("INTO")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        columns: tuple[str, ...] = ()
        if self._match_kind(TokenKind.LPAREN):
            columns = self._parse_column_name_list()
        self._expect_keyword("VALUES")
        values = self._parse_value_rows()
        self._error_if_not_eof()
        return InsertStatement(table=table, columns=columns, values=values)

    def _parse_column_name_list(self) -> tuple[str, ...]:
        names = [self._expect_kind(TokenKind.IDENTIFIER).value]
        while self._match_comma():
            names.append(self._expect_kind(TokenKind.IDENTIFIER).value)
        self._expect_kind(TokenKind.RPAREN)
        return tuple(names)

    def _parse_create_table(self) -> CreateTableStatement:
        self._expect_keyword("TABLE")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        self._expect_kind(TokenKind.LPAREN)
        columns = [self._parse_column_def()]
        while self._match_comma():
            columns.append(self._parse_column_def())
        self._expect_kind(TokenKind.RPAREN)
        engine = "HEAP"
        if self._match_keyword("ENGINE"):
            engine = self._expect_ident_or_keyword()
        self._error_if_not_eof()
        return CreateTableStatement(table=table, columns=tuple(columns), engine=engine)

    def _parse_create_index(self) -> CreateIndexStatement:
        index_name = self._expect_kind(TokenKind.IDENTIFIER).value
        self._expect_keyword("ON")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        self._expect_kind(TokenKind.LPAREN)
        column = self._expect_kind(TokenKind.IDENTIFIER).value
        self._expect_kind(TokenKind.RPAREN)
        index_type = "BTREE"
        if self._match_keyword("TYPE"):
            index_type = self._expect_ident_or_keyword()
        self._error_if_not_eof()
        return CreateIndexStatement(
            index_name=index_name,
            table=table,
            column=column,
            index_type=index_type,
        )

    def _parse_column_def(self) -> ColumnDef:
        name = self._expect_kind(TokenKind.IDENTIFIER).value
        type_token = self._expect_kind(TokenKind.IDENTIFIER)
        type_name = self._column_type(type_token.value, type_token.position)
        length = None
        if type_name is ColumnType.VARCHAR:
            self._expect_kind(TokenKind.LPAREN)
            length_token = self._expect_kind(TokenKind.NUMBER)
            if "." in length_token.value:
                raise QueryParseError(
                    f"VARCHAR length must be an integer at position {length_token.position}"
                )
            length = int(length_token.value)
            self._expect_kind(TokenKind.RPAREN)
        return ColumnDef(name=name, type_name=type_name, length=length)

    @staticmethod
    def _column_type(raw: str, position: int) -> ColumnType:
        try:
            return ColumnType(raw.upper())
        except ValueError:
            raise QueryParseError(
                f"unknown column type {raw.upper()!r} at position {position}"
            ) from None

    def _parse_value_rows(self) -> tuple[tuple[Expr, ...], ...]:
        rows = [self._parse_value_row()]
        while self._match_comma():
            rows.append(self._parse_value_row())
        return tuple(rows)

    def _parse_value_row(self) -> tuple[Expr, ...]:
        self._expect_kind(TokenKind.LPAREN)
        values = [self._parse_expression()]
        while self._match_comma():
            values.append(self._parse_expression())
        self._expect_kind(TokenKind.RPAREN)
        return tuple(values)

    def _parse_select(self) -> SelectStatement:
        distinct = self._match_keyword("DISTINCT")
        columns = self._parse_projection()
        self._expect_keyword("FROM")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        where = None
        if self._match_keyword("WHERE"):
            where = self._parse_boolean_expression()
        group_by: tuple[Expr, ...] = ()
        if self._match_keyword("GROUP"):
            self._expect_keyword("BY")
            group_by = self._parse_group_by()
        order_by: tuple[OrderByItem, ...] = ()
        if self._match_keyword("ORDER"):
            self._expect_keyword("BY")
            order_by = self._parse_order_by()
        self._error_if_not_eof()
        return SelectStatement(
            columns=columns,
            table=table,
            where=where,
            group_by=group_by,
            order_by=order_by,
            distinct=distinct,
        )

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

    def _parse_group_by(self) -> tuple[Expr, ...]:
        items = [self._parse_expression()]
        while self._match_comma():
            items.append(self._parse_expression())
        return tuple(items)

    def _parse_order_by(self) -> tuple[OrderByItem, ...]:
        items = [self._parse_order_by_item()]
        while self._match_comma():
            items.append(self._parse_order_by_item())
        return tuple(items)

    def _parse_order_by_item(self) -> OrderByItem:
        expr = self._parse_expression()
        ascending = True
        if self._match_keyword("DESC"):
            ascending = False
        elif self._match_keyword("ASC"):
            ascending = True
        return OrderByItem(expr=expr, ascending=ascending)

    def _parse_expression(self) -> Expr:
        """Parse any expression used in projections and clause keys.

        Comparisons are optional here: a bare column reference is a valid
        projection term.
        """
        return self._parse_or()

    def _parse_boolean_expression(self) -> Expr:
        """Parse a WHERE predicate, which must be a boolean-capable node."""
        expr = self._parse_expression()
        if not self._is_boolean_expression(expr):
            token = self._peek()
            raise QueryParseError(
                f"expected boolean expression at position {token.position}, got {token.value!r}"
            )
        return expr

    @staticmethod
    def _is_boolean_expression(expr: Expr) -> bool:
        if isinstance(expr, (CompareExpr, BetweenExpr, InExpr, LikeExpr)):
            return True
        if isinstance(expr, Literal) and isinstance(expr.value, bool):
            return True
        if isinstance(expr, NotExpr):
            return _Parser._is_boolean_expression(expr.operand)
        if isinstance(expr, LogicalExpr):
            return _Parser._is_boolean_expression(expr.left) and _Parser._is_boolean_expression(
                expr.right
            )
        return False

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

    def _check_keyword_in(self, keywords: set[str]) -> bool:
        token = self._peek()
        return token.kind is TokenKind.KEYWORD and token.value in keywords

    def _checks_lparen_next(self) -> bool:
        if self._pos + 1 >= len(self._tokens):
            return False
        return self._tokens[self._pos + 1].kind is TokenKind.LPAREN

    def _parse_function_call(self) -> FunctionExpr:
        name = self._advance().value
        self._expect_kind(TokenKind.LPAREN)
        distinct = self._match_keyword("DISTINCT")
        arg = None
        if self._check_kind(TokenKind.STAR):
            star_token = self._advance()
            if name != "COUNT" or distinct:
                raise QueryParseError(
                    f"only COUNT(*) is supported at position {star_token.position}"
                )
        else:
            arg = self._parse_expression()
        self._expect_kind(TokenKind.RPAREN)
        return FunctionExpr(name=name, arg=arg, distinct=distinct)

    def _parse_primary(self) -> Expr:
        token = self._peek()

        if self._match_kind(TokenKind.NUMBER):
            return Literal(_number_value(token.value))

        if self._match_kind(TokenKind.STRING):
            return Literal(token.value)

        if self._match_keyword("TRUE"):
            return Literal(True)

        if self._match_keyword("FALSE"):
            return Literal(False)

        if self._check_keyword_in(_AGGREGATES) and self._checks_lparen_next():
            return self._parse_function_call()

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
