import math

import pytest

from engine.indexes.inverted.scoring import bm25_term, cosine, idf, validate_bm25


def test_unsmoothed_idf_and_zero_vectors():
    assert idf(3, 2) == pytest.approx(math.log(1.5))
    assert idf(3, 3) == 0.0
    assert idf(0, 0) == idf(3, 0) == 0.0
    assert cosine(0.0, 0.0, 1.0) == cosine(0.0, 1.0, 0.0) == 0.0
    assert cosine(11.0, 5.0, math.sqrt(5.0)) == pytest.approx(11 / (5 * math.sqrt(5)))


def test_bm25_manual_calculation_and_length_normalization():
    expected = math.log(1.6) * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    score = bm25_term(2, n=3, df=2, doc_len=3, avgdl=2.0)
    assert score == pytest.approx(expected)
    shorter = bm25_term(2, n=3, df=2, doc_len=1, avgdl=2.0)
    assert shorter > score
    assert bm25_term(0, n=0, df=0, doc_len=0, avgdl=0.0) == 0.0
    assert math.isfinite(bm25_term(2, n=3, df=2, doc_len=3, avgdl=2, k1=1e308))
    assert bm25_term(2, n=3, df=2, doc_len=3, avgdl=2, k1=1e-320) > 0.0


@pytest.mark.parametrize(
    "k1,b",
    [
        (0, 0.75),
        (-1, 0.75),
        (1.2, -1),
        (1.2, 2),
        (math.nan, 0.75),
        (1.2, math.inf),
        (True, 0.75),
        (10**1000, 0.75),
        (1.2, 10**1000),
    ],
)
def test_invalid_bm25_parameters(k1, b):
    with pytest.raises(ValueError):
        validate_bm25(k1, b)


@pytest.mark.parametrize("n,df", [(2, 3), (-1, 0), (True, 1), (1, -1)])
def test_invalid_document_frequencies(n, df):
    with pytest.raises(ValueError):
        idf(n, df)
