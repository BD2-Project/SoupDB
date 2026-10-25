"""Metrics of 2.4.2 against values computed by hand."""

import math

import pytest

from engine.indexes.vector.metrics import (
    COSINE,
    EUCLIDEAN,
    INNER_PRODUCT,
    cosine_similarity,
    dot,
    euclidean_distance,
    norm,
    score,
    score_fn,
)


def test_euclidean_distance_of_3_4_5_triangle() -> None:
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == 5.0


def test_dot_product() -> None:
    assert dot([1.0, 2.0, 3.0], [4.0, -5.0, 6.0]) == 12.0


def test_norm() -> None:
    assert norm([3.0, 4.0]) == 5.0


def test_cosine_of_parallel_orthogonal_and_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 5.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_by_hand() -> None:
    # (1,2,2)·(2,0,1) = 4; |a| = 3; |b| = sqrt(5)
    assert cosine_similarity([1.0, 2.0, 2.0], [2.0, 0.0, 1.0]) == pytest.approx(
        4 / (3 * math.sqrt(5))
    )


def test_cosine_with_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_scores_are_higher_is_better() -> None:
    query = [0.0, 0.0]
    near, far = [1.0, 0.0], [3.0, 4.0]

    assert score(EUCLIDEAN, query, near) == pytest.approx(0.5)
    assert score(EUCLIDEAN, query, far) == pytest.approx(1 / 6)
    assert score(EUCLIDEAN, near, near) == 1.0
    assert score(INNER_PRODUCT, [1.0, 2.0], [3.0, 4.0]) == 11.0
    assert score(COSINE, [1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_fn("manhattan")
