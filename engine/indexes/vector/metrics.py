"""Distance and similarity metrics of 2.4.2, in pure Python.

Every metric is exposed as a score where higher is better, so the indexes rank
with one rule regardless of the metric. The norm of each stored vector is
precomputed at insert time, so cosine costs one dot product per candidate.
"""

import math
from collections.abc import Callable, Sequence
from operator import mul

EUCLIDEAN = "euclidean"
COSINE = "cosine"
INNER_PRODUCT = "inner_product"
METRICS = (EUCLIDEAN, COSINE, INNER_PRODUCT)

Vector = Sequence[float]

#: score(query, query_norm, vector, vector_norm) -> float, higher is better.
ScoreFn = Callable[[Vector, float, Vector, float], float]

# math.sumprod está en C desde 3.12; en 3.11 se cae a map/sum, más lento pero igual de exacto.
_sumprod = getattr(math, "sumprod", None)


def dot(a: Vector, b: Vector) -> float:
    if _sumprod is not None:
        return _sumprod(a, b)
    return sum(map(mul, a, b))


def norm(vector: Vector) -> float:
    return math.hypot(*vector)


def euclidean_distance(a: Vector, b: Vector) -> float:
    return math.dist(a, b)


def cosine_similarity(a: Vector, b: Vector) -> float:
    norms = norm(a) * norm(b)
    return dot(a, b) / norms if norms else 0.0


def _euclidean_score(query: Vector, _qn: float, vector: Vector, _vn: float) -> float:
    # 1/(1+d) en vez de -d: el vector idéntico puntúa 1.0, que es lo que muestra SIMILARITY_SCORE().
    return 1.0 / (1.0 + math.dist(query, vector))


def _cosine_score(query: Vector, query_norm: float, vector: Vector, vector_norm: float) -> float:
    norms = query_norm * vector_norm
    return dot(query, vector) / norms if norms else 0.0


def _inner_product_score(query: Vector, _qn: float, vector: Vector, _vn: float) -> float:
    return dot(query, vector)


_SCORERS: dict[str, ScoreFn] = {
    EUCLIDEAN: _euclidean_score,
    COSINE: _cosine_score,
    INNER_PRODUCT: _inner_product_score,
}


def score_fn(metric: str) -> ScoreFn:
    """Resolve a metric name once, so the hot loop does not compare strings."""
    try:
        return _SCORERS[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}") from None


def score(metric: str, a: Vector, b: Vector) -> float:
    """Score two vectors with a metric; convenience for tests and callers outside hot loops."""
    return score_fn(metric)(a, norm(a), b, norm(b))
