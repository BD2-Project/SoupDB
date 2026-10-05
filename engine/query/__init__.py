"""Public API of the SQL query processing domain.

``execute_sql`` parses, plans and executes a SQL statement against a
duck-typed catalog (see ``engine.query.planner``) returning an internal
``ResultSet``.
"""

from typing import Any

from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from engine.query.resultset import ResultSet

__all__ = ["ResultSet", "execute_sql", "parse", "plan"]


def execute_sql(sql: str, catalog: Any) -> ResultSet:
    """Parse, plan and execute a SQL statement, returning a result set."""
    return execute(plan(parse(sql), catalog), catalog)
