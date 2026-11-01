"""Tests for C05 score binding and scalar evaluation.

Scores are read from a temporary FLOAT slot, not recalculated from an index.
"""

import math
from dataclasses import FrozenInstanceError

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import (
    BoundScoreExpr,
    ColumnRef,
    CompareExpr,
    Literal,
    ScoreExpr,
    SelectColumn,
    StarExpr,
)
from engine.query.evaluator import evaluate
from engine.query.ranked_binding import (
    append_score_schema,
    bind_score,
    visible_projections,
)

BASE_SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("contenido", ColumnType.TEXT),
)


def test_append_private_score_slot() -> None:
    schema = append_score_schema(BASE_SCHEMA)

    assert schema[:-1] == BASE_SCHEMA
    assert schema[-1].type_name is ColumnType.FLOAT
    assert "\x00" in schema[-1].name
    assert len(schema) == 3
    assert BASE_SCHEMA == (
        ColumnDef("id", ColumnType.INT),
        ColumnDef("contenido", ColumnType.TEXT),
    )


def test_appending_score_twice_is_rejected() -> None:
    with pytest.raises(QueryExecutionError, match="already exists"):
        append_score_schema(append_score_schema(BASE_SCHEMA))


def test_bind_score_uses_position_not_index() -> None:
    result = bind_score(ScoreExpr(kind="text"), position=2, kind="text")

    assert result == BoundScoreExpr(position=2, kind="text")


def test_binding_preserves_ordinary_expressions() -> None:
    expression = ColumnRef("contenido")

    assert bind_score(expression, position=2, kind="text") is expression


def test_bind_vector_score() -> None:
    assert bind_score(ScoreExpr(kind="vector"), position=2, kind="vector") == (
        BoundScoreExpr(position=2, kind="vector")
    )


@pytest.mark.parametrize("position", [-1, 1.5, True])
def test_binding_rejects_invalid_position(position: object) -> None:
    with pytest.raises(QueryExecutionError, match="position"):
        bind_score(ScoreExpr(), position=position, kind="text")


def test_binding_rejects_unknown_score_kind() -> None:
    with pytest.raises(QueryExecutionError, match="unsupported score kind"):
        bind_score(ScoreExpr(), position=2, kind="unknown")


def test_binding_rejects_kind_mismatch() -> None:
    with pytest.raises(QueryExecutionError, match="cannot be used"):
        bind_score(ScoreExpr(kind="vector"), position=2, kind="text")


def test_binding_rejects_score_nested_in_an_expression() -> None:
    expression = CompareExpr(ScoreExpr(), ">", Literal(0.5))

    with pytest.raises(QueryExecutionError, match="standalone"):
        bind_score(expression, position=2, kind="text")


def test_binding_rejects_already_bound_score() -> None:
    with pytest.raises(QueryExecutionError, match="standalone"):
        bind_score(BoundScoreExpr(2, "text"), position=2, kind="text")


def test_plain_star_projects_only_original_columns() -> None:
    projections = visible_projections((), BASE_SCHEMA, position=2, kind="text")

    assert projections == (
        SelectColumn(ColumnRef("id")),
        SelectColumn(ColumnRef("contenido")),
    )


def test_mixed_star_and_score_projects_only_visible_columns() -> None:
    projections = visible_projections(
        (SelectColumn(StarExpr()), SelectColumn(ScoreExpr(), alias="relevancia")),
        BASE_SCHEMA,
        position=2,
        kind="text",
    )

    assert projections == (
        SelectColumn(ColumnRef("id")),
        SelectColumn(ColumnRef("contenido")),
        SelectColumn(BoundScoreExpr(2, "text"), alias="relevancia"),
    )


def test_score_only_projection() -> None:
    projections = visible_projections(
        (SelectColumn(ScoreExpr(), alias="relevancia"),),
        BASE_SCHEMA,
        position=2,
        kind="text",
    )

    assert projections == (SelectColumn(BoundScoreExpr(2, "text"), alias="relevancia"),)


def test_mixed_star_rejects_duplicate_column_alias() -> None:
    with pytest.raises(QueryExecutionError, match="conflicts"):
        visible_projections(
            (SelectColumn(StarExpr()), SelectColumn(ScoreExpr(), alias="id")),
            BASE_SCHEMA,
            position=2,
            kind="text",
        )


@pytest.mark.parametrize(
    "columns",
    [
        (SelectColumn(StarExpr()),),
        (SelectColumn(ColumnRef("id")), SelectColumn(StarExpr())),
    ],
)
def test_invalid_star_position_is_rejected(
    columns: tuple[SelectColumn, ...],
) -> None:
    with pytest.raises(QueryExecutionError, match="invalid mixed"):
        visible_projections(columns, BASE_SCHEMA, position=2, kind="text")


def test_projection_rejects_non_appended_score_position() -> None:
    with pytest.raises(QueryExecutionError, match="immediately follow"):
        visible_projections((), BASE_SCHEMA, position=0, kind="text")


def test_evaluate_bound_score_uses_exact_stored_float() -> None:
    schema = append_score_schema(BASE_SCHEMA)
    row = (7, "documento", 0.75)

    assert evaluate(BoundScoreExpr(2, "text"), row, schema) == 0.75
    assert evaluate(ColumnRef("id"), row, schema) == 7


def test_evaluate_negative_score_is_valid() -> None:
    schema = append_score_schema(BASE_SCHEMA)

    assert evaluate(BoundScoreExpr(2, "vector"), (1, "imagen", -0.5), schema) == -0.5


@pytest.mark.parametrize("value", [None, 0, True, "0.5", math.nan, math.inf, -math.inf])
def test_evaluate_rejects_invalid_scores(value: object) -> None:
    schema = append_score_schema(BASE_SCHEMA)

    with pytest.raises(QueryExecutionError, match="finite FLOAT"):
        evaluate(BoundScoreExpr(2, "text"), (1, "texto", value), schema)


def test_evaluate_rejects_missing_score_value() -> None:
    schema = append_score_schema(BASE_SCHEMA)

    with pytest.raises(QueryExecutionError, match="missing from row"):
        evaluate(BoundScoreExpr(2, "text"), (1, "texto"), schema)


@pytest.mark.parametrize("position", [0, 3, -1, True])
def test_evaluate_rejects_wrong_score_position(position: object) -> None:
    schema = append_score_schema(BASE_SCHEMA)

    with pytest.raises(QueryExecutionError, match="invalid bound score position"):
        evaluate(BoundScoreExpr(position, "text"), (1, "texto", 0.9), schema)


def test_evaluate_rejects_score_without_private_slot() -> None:
    with pytest.raises(QueryExecutionError, match="internal FLOAT"):
        evaluate(
            BoundScoreExpr(1, "text"),
            (1, 0.9),
            (ColumnDef("id", ColumnType.INT), ColumnDef("score", ColumnType.FLOAT)),
        )


def test_evaluate_rejects_score_with_wrong_slot_type() -> None:
    internal_name = append_score_schema(BASE_SCHEMA)[-1].name
    schema = (*BASE_SCHEMA, ColumnDef(internal_name, ColumnType.INT))

    with pytest.raises(QueryExecutionError, match="internal FLOAT"):
        evaluate(BoundScoreExpr(2, "text"), (1, "texto", 1.0), schema)


def test_unbound_score_is_not_evaluable() -> None:
    with pytest.raises(QueryExecutionError, match="unsupported expression"):
        evaluate(ScoreExpr(), (1, "texto"), BASE_SCHEMA)


def test_bound_score_is_immutable() -> None:
    expr = BoundScoreExpr(2, "text")

    with pytest.raises(FrozenInstanceError):
        expr.position = 0
