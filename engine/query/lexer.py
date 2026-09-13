"""SQL tokenizer.

Turns a SQL string into a list of :class:`engine.query.tokens.Token`. The
lexer is keyword-aware (keywords are normalized to uppercase), recognizes
single-quoted string literals, integer and float numbers, and the comparison
operators required by the query language.
"""

from engine.query.errors import QueryParseError
from engine.query.tokens import Token, TokenKind

_KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "BETWEEN",
    "IN",
    "LIKE",
    "ORDER",
    "BY",
    "GROUP",
    "ASC",
    "DESC",
    "DISTINCT",
    "INSERT",
    "INTO",
    "VALUES",
    "DELETE",
    "CREATE",
    "TABLE",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
}

_OPERATORS = {">=", "<=", "<>", "!=", ">", "<", "="}
_DIGITS = set("0123456789")
_WHITESPACE = {" ", "\t", "\r", "\n"}
_PUNCTUATION = {
    "*": TokenKind.STAR,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    ",": TokenKind.COMMA,
    ";": TokenKind.SEMICOLON,
}


def _is_identifier_start(char: str) -> bool:
    return char == "_" or char.isalpha()


def _is_identifier_part(char: str) -> bool:
    return char == "_" or char.isalnum()


def _scan_string(sql: str, i: int) -> tuple[str, int]:
    """Scan a single-quoted string; returns its content and next index."""
    i += 1
    chars: list[str] = []
    while i < len(sql):
        char = sql[i]
        if char == "'":
            return "".join(chars), i + 1
        chars.append(char)
        i += 1
    raise QueryParseError(f"unterminated string literal at position {i}")


def _scan_identifier(sql: str, i: int) -> tuple[str, int]:
    chars: list[str] = []
    while i < len(sql) and _is_identifier_part(sql[i]):
        chars.append(sql[i])
        i += 1
    return "".join(chars), i


def _scan_number(sql: str, i: int) -> tuple[str, int]:
    chars: list[str] = []
    while i < len(sql) and (sql[i] in _DIGITS or sql[i] == "."):
        chars.append(sql[i])
        i += 1
    return "".join(chars), i


def tokenize(sql: str) -> list[Token]:
    """Split ``sql`` into tokens ending with an EOF token."""
    tokens: list[Token] = []
    i = 0
    while i < len(sql):
        char = sql[i]
        if char in _WHITESPACE:
            i += 1
            continue

        if char == "'":
            content, i = _scan_string(sql, i)
            tokens.append(Token(TokenKind.STRING, content, i - len(content) - 2))
            continue

        if char in _DIGITS:
            start = i
            number, i = _scan_number(sql, i)
            tokens.append(Token(TokenKind.NUMBER, number, start))
            continue

        two = sql[i : i + 2]
        if two in _OPERATORS:
            tokens.append(Token(TokenKind.OPERATOR, two, i))
            i += 2
            continue

        if char in _OPERATORS:
            tokens.append(Token(TokenKind.OPERATOR, char, i))
            i += 1
            continue

        if char in _PUNCTUATION:
            tokens.append(Token(_PUNCTUATION[char], char, i))
            i += 1
            continue

        if _is_identifier_start(char):
            start = i
            word, i = _scan_identifier(sql, i)
            upper = word.upper()
            kind = TokenKind.KEYWORD if upper in _KEYWORDS else TokenKind.IDENTIFIER
            tokens.append(Token(kind, upper if kind is TokenKind.KEYWORD else word, start))
            continue

        raise QueryParseError(f"unexpected character {char!r} at position {i}")

    tokens.append(Token(TokenKind.EOF, "", len(sql)))
    return tokens
