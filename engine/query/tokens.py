"""Token kinds and token dataclass produced by the SQL lexer."""

from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    """Classification of a single SQL token."""

    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    NUMBER = "number"
    STRING = "string"
    OPERATOR = "operator"
    STAR = "star"
    LPAREN = "lparen"
    RPAREN = "rparen"
    COMMA = "comma"
    SEMICOLON = "semicolon"
    EOF = "eof"


@dataclass(frozen=True)
class Token:
    """A lexical unit with its position in the source string.

    ``value`` holds the raw source text, except keywords which are normalized
    to uppercase and string literals which store the unquoted content.
    ``position`` is the offset of the first character.
    """

    kind: TokenKind
    value: str
    position: int
