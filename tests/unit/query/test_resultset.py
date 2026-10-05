"""Tests for the internal ResultSet value object."""

from engine.common.schema import ColumnDef, ColumnType
from engine.query.resultset import ResultSet

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
)
ROWS = ((1, "A"), (2, "B"))


def test_resultset_holds_columns_and_rows() -> None:
    result = ResultSet(columns=PAPERS, rows=ROWS)
    assert result.columns == PAPERS
    assert result.rows == ROWS
    assert result.affected == 0


def test_resultset_defaults_empty() -> None:
    result = ResultSet(columns=())
    assert result.rows == ()
    assert result.affected == 0


def test_resultset_affected_count() -> None:
    result = ResultSet(columns=(), affected=3)
    assert result.rows == ()
    assert result.affected == 3


def test_resultset_is_immutable() -> None:
    result = ResultSet(columns=PAPERS, rows=ROWS)
    assert result.columns[0].name == "id"
    assert result.rows[0] == (1, "A")
