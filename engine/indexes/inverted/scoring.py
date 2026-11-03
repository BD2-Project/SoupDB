"""Raw-TF TF-IDF cosine and positive-IDF Okapi BM25 formulas.

TF-IDF weight = tf * log(N/df). Norms use the entire document vector; zero
vectors score zero. BM25 uses log(1 + (N-df+0.5)/(df+0.5)) and raw query
frequency as a multiplier. Matching is determined by postings, never score.
"""

import math


def idf(n: int, df: int) -> float:
    """Return unsmoothed IDF; an absent term has no weight."""
    if type(n) is not int or type(df) is not int or not 0 <= df <= n:
        raise ValueError("expected 0 <= df <= N")
    return math.log(n / df) if df else 0.0


def cosine(dot: float, query_norm: float, document_norm: float) -> float:
    """Cosine for nonnegative TF-IDF vectors, defining zero-vector behavior."""
    if query_norm == 0.0 or document_norm == 0.0:
        return 0.0
    return float(min(1.0, max(0.0, dot / query_norm / document_norm)))


def validate_bm25(k1: float, b: float) -> None:
    for name, value in (("k1", k1), ("b", b)):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"BM25 {name} must be a finite number")
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")


def bm25_term(
    tf: int, *, n: int, df: int, doc_len: int, avgdl: float,
    k1: float = 1.2, b: float = 0.75,
) -> float:
    """Score one matching term; the caller validates shared BM25 parameters."""
    if tf == 0 or avgdl == 0.0:
        return 0.0
    inverse_frequency = math.log1p((n - df + 0.5) / (df + 0.5))
    normalization = 1.0 - b + b * doc_len / avgdl
    # Equivalent to tf*(k1+1)/(tf+k1*normalization), avoiding large-k1 overflow.
    if k1 >= 1.0:
        saturation = (1.0 + 1.0 / k1) / (1.0 / k1 + normalization / tf)
    else:
        saturation = (1.0 + k1) / (1.0 + k1 * normalization / tf)
    return float(inverse_frequency * saturation)
