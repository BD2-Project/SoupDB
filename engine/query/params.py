"""Bind prepared-statement parameters into a parsed AST.

Parameters are replaced before planning, so the parser runs once per prepared
statement and the binder never interpolates SQL text (no injection surface).
"""

from dataclasses import fields, is_dataclass, replace
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.query.ast import Literal, Parameter


def bind_parameters(node: Any, params: tuple[object, ...]) -> Any:
    """Return a copy of ``node`` with every ``$n`` replaced by its parameter."""
    if isinstance(node, Parameter):
        if not 1 <= node.index <= len(params):
            raise QueryExecutionError(
                f"prepared statement requires at least {node.index} parameter(s); got {len(params)}"
            )
        return Literal(params[node.index - 1])
    if isinstance(node, tuple):
        return tuple(bind_parameters(item, params) for item in node)
    if isinstance(node, list):
        return [bind_parameters(item, params) for item in node]
    if is_dataclass(node):
        return replace(
            node,
            **{
                field.name: bind_parameters(getattr(node, field.name), params)
                for field in fields(node)
            },
        )
    return node
