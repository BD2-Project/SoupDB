"""Recursive descent SQL parser producing an AST.

The parser consumes the token stream from :mod:`engine.query.lexer` and builds
:mod:`engine.query.ast` nodes. Entry point is :func:`parse`.

Expression precedence (low to high):

``OR`` < ``AND`` < ``NOT`` < comparison (``= <> != < <= > >=``,
``BETWEEN``, ``IN``, ``LIKE``) < additive (``+ -``) < multiplicative
(``* / %``) < primary. ``IS [NOT] NULL`` is a predicate suffix that binds at
the comparison level.
"""

from engine.query.ast import (
    BeginTransactionStatement,
    BetweenExpr,
    BinaryExpr,
    ColumnDef,
    ColumnRef,
    ColumnType,
    CompareExpr,
    CreateIndexStatement,
    CreateTableStatement,
    DeleteStatement,
    DistanceExpr,
    DropIndexStatement,
    DropTableStatement,
    EndTransactionStatement,
    ExplainStatement,
    Expr,
    FunctionExpr,
    HavingClause,
    InExpr,
    InsertStatement,
    IntersectsExpr,
    IsNullExpr,
    JoinClause,
    LikeExpr,
    LimitClause,
    Literal,
    LogicalExpr,
    NotExpr,
    OrderByItem,
    PointExpr,
    PolygonExpr,
    SelectColumn,
    SelectStatement,
    Statement,
    UpdateStatement,
)
from engine.query.errors import QueryParseError
from engine.query.lexer import tokenize
from engine.query.spatial_metrics import normalize_metric
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

    def __init__(self, tokens: list[Token], sql: str = "") -> None:
        self._tokens = tokens
        self._pos = 0
        self._sql = sql

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
        if self._match_keyword("UPDATE"):
            return self._parse_update()
        if self._match_keyword("BEGIN"):
            return self._parse_begin_transaction()
        if self._match_keyword("END"):
            return self._parse_end_transaction()
        if self._match_keyword("EXPLAIN"):
            return self._parse_explain()
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
        if self._match_keyword("DROP"):
            if self._match_keyword("INDEX"):
                return self._parse_drop_index()
            return self._parse_drop_table()
        raise QueryParseError(f"unsupported statement at position {self._peek().position}")

    def _parse_begin_transaction(self) -> BeginTransactionStatement:
        """Parse ``BEGIN`` and ``BEGIN TRANSACTION`` (the word is optional)."""
        self._match_keyword("TRANSACTION")
        self._error_if_not_eof()
        return BeginTransactionStatement()

    def _parse_end_transaction(self) -> EndTransactionStatement:
        """Parse ``END`` and ``END TRANSACTION`` (the word is optional)."""
        self._match_keyword("TRANSACTION")
        self._error_if_not_eof()
        return EndTransactionStatement()

    def _parse_update(self) -> UpdateStatement:
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        self._expect_keyword("SET")
        assignments = [self._parse_assignment()]
        while self._match_comma():
            assignments.append(self._parse_assignment())
        where = None
        if self._match_keyword("WHERE"):
            where = self._parse_boolean_expression()
        self._error_if_not_eof()
        return UpdateStatement(table=table, assignments=tuple(assignments), where=where)

    def _parse_assignment(self) -> tuple[str, Expr]:
        token = self._peek()
        if token.kind not in (TokenKind.IDENTIFIER, TokenKind.KEYWORD):
            raise QueryParseError(
                f"expected a column name at position {token.position}, got {token.value!r}"
            )
        column = self._advance().value
        eq = self._peek()
        if eq.kind is not TokenKind.OPERATOR or eq.value != "=":
            raise QueryParseError(f"expected '=' at position {eq.position}, got {eq.value!r}")
        self._advance()
        return column, self._parse_expression()

    def _parse_explain(self) -> ExplainStatement:
        analyze = self._match_keyword("ANALYZE")
        inner_sql = self._sql[self._peek().position :]
        statement = self.parse_statement()
        return ExplainStatement(sql=inner_sql, statement=statement, analyze=analyze)

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
        primary_key: str | None = None
        columns = []
        while True:
            column, is_primary = self._parse_column_def_with_constraints()
            if is_primary:
                if primary_key is not None:
                    raise QueryParseError(f"table {table!r} declares more than one PRIMARY KEY")
                primary_key = column.name
            columns.append(column)
            if not self._match_comma():
                break
        self._expect_kind(TokenKind.RPAREN)
        engine = "HEAP"
        if self._match_keyword("ENGINE"):
            engine = self._expect_ident_or_keyword()
        self._error_if_not_eof()
        return CreateTableStatement(
            table=table,
            columns=tuple(columns),
            engine=engine,
            primary_key=primary_key,
        )

    def _parse_column_def_with_constraints(self) -> tuple[ColumnDef, bool]:
        """Una definición de columna y si lleva ``PRIMARY KEY``."""
        column = self._parse_column_def()
        if not self._match_keyword("PRIMARY"):
            return column, False
        self._expect_keyword("KEY")
        return column, True

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

    def _parse_drop_table(self) -> DropTableStatement:
        self._expect_keyword("TABLE")
        table = self._expect_kind(TokenKind.IDENTIFIER).value
        self._error_if_not_eof()
        return DropTableStatement(table=table)

    def _parse_drop_index(self) -> DropIndexStatement:
        index_name = self._expect_kind(TokenKind.IDENTIFIER).value
        self._error_if_not_eof()
        return DropIndexStatement(index_name=index_name)

    def _parse_column_def(self) -> ColumnDef:
        name = self._expect_kind(TokenKind.IDENTIFIER).value
        token = self._peek()
        if token.kind not in (TokenKind.IDENTIFIER, TokenKind.KEYWORD):
            raise QueryParseError(
                f"expected a column type at position {token.position}, got {token.value!r}"
            )
        self._advance()
        type_name = self._column_type(token.value, token.position)
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
        joins = self._parse_joins()
        where = None
        if self._match_keyword("WHERE"):
            where = self._parse_boolean_expression()
        group_by: tuple[Expr, ...] = ()
        if self._match_keyword("GROUP"):
            self._expect_keyword("BY")
            group_by = self._parse_group_by()
        having = None
        if self._match_keyword("HAVING"):
            having = HavingClause(self._parse_boolean_expression())
        order_by: tuple[OrderByItem, ...] = ()
        if self._match_keyword("ORDER"):
            self._expect_keyword("BY")
            order_by = self._parse_order_by()
        limit = self._parse_limit()
        self._error_if_not_eof()
        return SelectStatement(
            columns=columns,
            table=table,
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            distinct=distinct,
            joins=joins,
        )

    def _parse_joins(self) -> tuple[JoinClause, ...]:
        joins: list[JoinClause] = []
        while True:
            if self._match_keyword("JOIN"):
                pass
            elif self._match_keyword("INNER"):
                self._expect_keyword("JOIN")
            else:
                break
            table = self._expect_kind(TokenKind.IDENTIFIER).value
            self._expect_keyword("ON")
            on = self._parse_boolean_expression()
            joins.append(JoinClause(table=table, on=on))
        return tuple(joins)

    def _parse_limit(self) -> LimitClause | None:
        if not self._match_keyword("LIMIT"):
            return None
        limit = self._expect_positive_number("LIMIT")
        offset = None
        if self._match_keyword("OFFSET"):
            offset = self._expect_positive_number("OFFSET")
        return LimitClause(limit=limit, offset=offset)

    def _expect_positive_number(self, clause: str) -> Literal:
        token = self._peek()
        if token.kind is not TokenKind.NUMBER:
            raise QueryParseError(
                f"expected a positive integer for {clause} at position {token.position}, "
                f"got {token.value!r}"
            )
        if "." in token.value or token.value.startswith("-"):
            raise QueryParseError(
                f"{clause} must be a positive integer at position {token.position}"
            )
        self._advance()
        return Literal(int(token.value))

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
        if isinstance(
            expr, (CompareExpr, BetweenExpr, InExpr, LikeExpr, IsNullExpr, IntersectsExpr)
        ):
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
        value = self._parse_arithmetic()
        token = self._peek()

        if self._match_keyword("IS"):
            negated = self._match_keyword("NOT")
            self._expect_kind(TokenKind.NULL)
            return IsNullExpr(value, negated=negated)

        if token.kind is TokenKind.OPERATOR and token.value in _COMPARISON_OPS:
            self._advance()
            right = self._parse_arithmetic()
            return CompareExpr(value, token.value, right)

        if self._match_keyword("BETWEEN"):
            lo = self._parse_arithmetic()
            self._expect_keyword("AND")
            hi = self._parse_arithmetic()
            return BetweenExpr(value, lo, hi)

        if self._match_keyword("IN"):
            self._expect_kind(TokenKind.LPAREN)
            items = [self._parse_arithmetic()]
            while self._match_comma():
                items.append(self._parse_arithmetic())
            self._expect_kind(TokenKind.RPAREN)
            return InExpr(value, tuple(items))

        if self._match_keyword("LIKE"):
            pattern = self._parse_arithmetic()
            return LikeExpr(value, pattern)

        return value

    def _check_arithmetic_operator(self, ops: tuple[str, ...], star: bool = False) -> bool:
        token = self._peek()
        if token.kind is TokenKind.OPERATOR and token.value in ops:
            return True
        return star and token.kind is TokenKind.STAR

    def _parse_arithmetic(self) -> Expr:
        left = self._parse_term()
        while self._check_arithmetic_operator(("+", "-")):
            op = self._advance().value
            right = self._parse_term()
            left = BinaryExpr(left, op, right)
        return left

    def _parse_term(self) -> Expr:
        left = self._parse_primary()
        while self._check_arithmetic_operator(("*", "/", "%"), star=True):
            token = self._advance()
            op = "*" if token.kind is TokenKind.STAR else token.value
            right = self._parse_primary()
            left = BinaryExpr(left, op, right)
        return left

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

    def _parse_point_expr(self) -> PointExpr:
        """``POINT(latitud, longitud)``, el orden que fija el enunciado (2.2.3).

        El intercambio ocurre acá y en ningún otro lado: hacia adentro el motor
        trabaja siempre con ``Point(x=longitud, y=latitud)``.
        """
        self._advance()
        self._expect_kind(TokenKind.LPAREN)
        latitude = self._parse_expression()
        self._expect_kind(TokenKind.COMMA)
        longitude = self._parse_expression()
        self._expect_kind(TokenKind.RPAREN)
        return PointExpr(x=longitude, y=latitude)

    def _parse_distance_expr(self) -> DistanceExpr:
        self._advance()
        self._expect_kind(TokenKind.LPAREN)
        left = self._parse_expression()
        self._expect_kind(TokenKind.COMMA)
        right = self._parse_expression()
        metric = "euclidean"
        if self._match_comma():
            metric = self._parse_distance_metric()
        self._expect_kind(TokenKind.RPAREN)
        return DistanceExpr(left=left, right=right, metric=metric)

    def _parse_distance_metric(self) -> str:
        """Third argument of ``distance``: a string literal naming the metric.

        The name is known at parse time, so an unknown metric is rejected here
        (as a ``QueryParseError``) instead of failing later at evaluation time,
        together with the rest of the literal syntax. The canonical lowercase
        name is stored in the AST.
        """
        token = self._peek()
        if token.kind is not TokenKind.STRING:
            raise QueryParseError(
                "distance() metric must be a string literal at position "
                f"{token.position}, got {token.value!r}"
            )
        try:
            return normalize_metric(self._advance().value)
        except ValueError as exc:
            raise QueryParseError(f"{exc} at position {token.position}") from exc

    def _parse_polygon_expr(self) -> PolygonExpr:
        self._advance()
        self._expect_kind(TokenKind.LPAREN)
        vertices: list[Expr] = []
        self._expect_kind(TokenKind.LPAREN)
        x = self._parse_expression()
        self._expect_kind(TokenKind.COMMA)
        y = self._parse_expression()
        self._expect_kind(TokenKind.RPAREN)
        vertices.append(PointExpr(x=x, y=y))
        while self._match_comma():
            self._expect_kind(TokenKind.LPAREN)
            x = self._parse_expression()
            self._expect_kind(TokenKind.COMMA)
            y = self._parse_expression()
            self._expect_kind(TokenKind.RPAREN)
            vertices.append(PointExpr(x=x, y=y))
        self._expect_kind(TokenKind.RPAREN)
        return PolygonExpr(vertices=tuple(vertices))

    def _parse_intersects_expr(self) -> IntersectsExpr:
        self._advance()
        self._expect_kind(TokenKind.LPAREN)
        left = self._parse_expression()
        self._expect_kind(TokenKind.COMMA)
        right = self._parse_expression()
        self._expect_kind(TokenKind.RPAREN)
        return IntersectsExpr(left=left, right=right)

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

        if self._match_kind(TokenKind.NULL):
            return Literal(None)

        if self._check_keyword_in(_AGGREGATES) and self._checks_lparen_next():
            return self._parse_function_call()

        if self._check_keyword("POINT") and self._checks_lparen_next():
            return self._parse_point_expr()

        if self._check_keyword("DISTANCE") and self._checks_lparen_next():
            return self._parse_distance_expr()

        if self._check_keyword("POLYGON") and self._checks_lparen_next():
            return self._parse_polygon_expr()

        if self._check_keyword("INTERSECTS") and self._checks_lparen_next():
            return self._parse_intersects_expr()

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
    parser = _Parser(tokenize(sql), sql)
    statement = parser.parse_statement()
    parser._error_if_not_eof()
    return statement
