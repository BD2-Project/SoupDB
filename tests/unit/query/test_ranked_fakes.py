"""SQL integration fake tests, independent of the real ranking algorithms."""

import pytest

from engine.common.rid import RID
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


def test_text_methods_use_separate_scripted_scores() -> None:
    index = ScriptedRankedIndex("text")
    query = "base de datos"
    first, second = RID(0, 1), RID(0, 2)
    index.set_results(query, [(first, 3.0), (second, 1.0)], method="bm25")
    index.set_results(query, [(second, 0.8), (first, 0.2)], method="tf_idf")

    assert index.top_k(query, 2, method="bm25") == [(first, 3.0), (second, 1.0)]
    assert index.top_k(query, 2, method="tf_idf") == [(second, 0.8), (first, 0.2)]
    assert index.calls == [
        (query, 2, {"method": "bm25"}),
        (query, 2, {"method": "tf_idf"}),
    ]


def test_text_zero_scores_and_duplicate_rids_are_preserved() -> None:
    index = ScriptedRankedIndex("text")
    hits = [(RID(1, 0), 0.0), (RID(1, 0), 0.0)]
    index.set_results("sin coincidencias", hits, method="bm25")

    assert index.top_k("sin coincidencias", 10, method="bm25") == hits
    # The SQL operator, not this scripted fake, validates provider inconsistencies.


def test_zero_k_and_larger_k_respect_the_scripted_list() -> None:
    index = ScriptedRankedIndex("text")
    hits = [(RID(1, 0), 2.0), (RID(1, 1), 1.0)]
    index.set_results("consulta", hits, method="bm25")

    assert index.top_k("consulta", 0, method="bm25") == []
    assert index.top_k("consulta", 1, method="bm25") == hits[:1]
    assert index.top_k("consulta", 99, method="bm25") == hits


def test_ivf_vector_query_and_options_are_forwarded() -> None:
    index = ScriptedRankedIndex("ivf")
    hits = [(RID(4, 2), 1.0), (RID(1, 1), 0.5)]
    index.set_results([1.0, 2.0], hits, metric="euclidean", nprobe=3)

    assert index.top_k([1.0, 2.0], 2, metric="euclidean", nprobe=3) == hits
    assert index.calls == [((1.0, 2.0), 2, {"metric": "euclidean", "nprobe": 3})]


def test_hnsw_accepts_negative_similarity_without_normalizing() -> None:
    index = ScriptedRankedIndex("hnsw")
    hits = [(RID(0, 0), 0.3), (RID(0, 1), -0.6)]
    index.set_results([1, 0], hits, metric="cosine", ef=24)

    assert index.top_k((1, 0), 2, metric="cosine", ef=24) == hits


def test_empty_hits_are_distinct_from_missing_script() -> None:
    index = ScriptedRankedIndex("text")
    index.set_results("desconocida", [], method="tf_idf")

    assert index.top_k("desconocida", 5, method="tf_idf") == []
    with pytest.raises(AssertionError, match="no scripted response"):
        index.top_k("no registrada", 5, method="tf_idf")


def test_scripted_provider_failure_is_raised() -> None:
    index = ScriptedRankedIndex("hnsw")
    index.set_error([0.0, 1.0], OSError("index unavailable"), metric="cosine")

    with pytest.raises(OSError, match="index unavailable"):
        index.top_k([0.0, 1.0], 2, metric="cosine")


@pytest.mark.parametrize("invalid_k", [-1, 1.5, True, "2"])
def test_invalid_k_is_rejected(invalid_k: object) -> None:
    index = ScriptedRankedIndex("text")

    with pytest.raises(ValueError, match="non-negative integer"):
        index.top_k("consulta", invalid_k, method="bm25")


@pytest.mark.parametrize("algorithm", ["random", "tf_idf", "brute_force"])
def test_unknown_algorithm_is_rejected(algorithm: str) -> None:
    with pytest.raises(ValueError, match="unsupported algorithm"):
        ScriptedRankedIndex(algorithm)


def test_invalid_options_and_query_types_are_rejected() -> None:
    text = ScriptedRankedIndex("text")
    vector = ScriptedRankedIndex("ivf")

    with pytest.raises(ValueError, match="unsupported search options"):
        text.top_k("consulta", 2, method="bm25", bogus=True)
    with pytest.raises(ValueError, match="requires method"):
        text.top_k("consulta", 2)
    with pytest.raises(TypeError, match="string query"):
        text.top_k([1.0], 2, method="bm25")
    with pytest.raises(TypeError, match="numeric sequence"):
        vector.top_k("imagen.jpg", 2, metric="cosine")
    with pytest.raises(ValueError, match="supported metric"):
        vector.top_k([1.0], 2, metric="haversine")
    with pytest.raises(ValueError, match="positive integer"):
        vector.top_k([1.0], 2, metric="cosine", nprobe=0)


def test_closed_index_cannot_be_searched() -> None:
    index = ScriptedRankedIndex("text")
    index.set_results("consulta", [(RID(0, 0), 1.0)], method="bm25")
    index.close()

    assert index.closed
    with pytest.raises(RuntimeError, match="closed"):
        index.top_k("consulta", 1, method="bm25")
