"""C06: ranked Volcano scans read RIDs and carry provider scores unchanged."""

from dataclasses import replace
from math import inf, nan

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import BoundScoreExpr, ColumnRef, SelectColumn
from engine.query.operators import Project
from engine.query.ranked_operators import InvertedIndexScan, RankedScan
from tests.fakes.fake_ranked_search import ScriptedRankedIndex
from tests.fakes.fake_storage import FakeFileOrganization

SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("contenido", ColumnType.TEXT),
)


@pytest.fixture
def search_data() -> tuple[FakeFileOrganization, ScriptedRankedIndex, RankedBinding, RID, RID]:
    storage = FakeFileOrganization()
    first = storage.insert(Record(data=encode_row((1, "primer documento"), SCHEMA)))
    second = storage.insert(Record(data=encode_row((2, "segundo documento"), SCHEMA)))
    index = ScriptedRankedIndex("text")
    binding = RankedBinding(
        name="idx_documentos_texto",
        table="documentos",
        column="contenido",
        algorithm="INVERTED",
        index=index,
        item_count=2,
        exact=True,
        exhaustive=True,
    )
    return storage, index, binding, first, second


def _scan(
    search_data: tuple[FakeFileOrganization, ScriptedRankedIndex, RankedBinding, RID, RID],
    *,
    request: int = 2,
    method: str = "bm25",
) -> InvertedIndexScan:
    storage, _index, binding, _first, _second = search_data
    return InvertedIndexScan(binding, storage, "consulta", SCHEMA, request=request, method=method)


def _drain(scan: RankedScan) -> list[tuple[object, ...]]:
    records = []
    while (record := scan.next()) is not None:
        records.append(decode_row(record.data, scan.schema))
    return records


def test_ranked_scan_keeps_provider_order_and_exact_scores(search_data) -> None:
    _storage, index, _binding, first, second = search_data
    index.set_results("consulta", [(second, 0.95), (first, 0.75)], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        assert _drain(scan) == [
            (2, "segundo documento", 0.95),
            (1, "primer documento", 0.75),
        ]
        assert scan.next() is None
        assert index.calls == [("consulta", 2, {"method": "bm25"})]
        assert scan.schema[-1].type_name is ColumnType.FLOAT
        assert scan.explain().rows == 2
    finally:
        scan.close()

    assert index.closed is False


def test_tf_idf_is_forwarded_to_provider(search_data) -> None:
    _storage, index, _binding, first, _second = search_data
    index.set_results("consulta", [(first, 1.5)], method="tf_idf")
    scan = _scan(search_data, method="tf_idf")

    scan.open()
    try:
        assert _drain(scan) == [(1, "primer documento", 1.5)]
    finally:
        scan.close()

    assert index.calls == [("consulta", 2, {"method": "tf_idf"})]


def test_empty_search_returns_no_rows(search_data) -> None:
    _storage, index, _binding, _first, _second = search_data
    index.set_results("consulta", [], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        assert scan.next() is None
        assert scan.explain().rows == 0
    finally:
        scan.close()


def test_zero_request_does_not_call_provider(search_data) -> None:
    _storage, index, _binding, _first, _second = search_data
    scan = _scan(search_data, request=0)

    scan.open()
    try:
        assert scan.next() is None
        assert index.calls == []
    finally:
        scan.close()


def test_provider_request_is_not_inferred_from_item_count(search_data) -> None:
    storage, index, binding, _first, _second = search_data
    unknown_count = replace(binding, item_count=None, exhaustive=False)
    index.set_results("consulta", [], method="bm25")
    scan = InvertedIndexScan(unknown_count, storage, "consulta", SCHEMA, request=20, method="bm25")

    scan.open()
    try:
        assert scan.next() is None
    finally:
        scan.close()

    assert index.calls == [("consulta", 20, {"method": "bm25"})]


def test_reopen_restarts_hits_and_does_not_close_shared_index(search_data) -> None:
    _storage, index, _binding, first, _second = search_data
    index.set_results("consulta", [(first, 0.5)], method="bm25")
    scan = _scan(search_data)

    for _ in range(2):
        scan.open()
        assert _drain(scan) == [(1, "primer documento", 0.5)]
        scan.close()

    assert len(index.calls) == 2
    assert index.closed is False


def test_next_before_open_raises(search_data) -> None:
    scan = _scan(search_data)
    with pytest.raises(QueryExecutionError, match="not open"):
        scan.next()


def test_double_open_raises_without_closing_provider(search_data) -> None:
    _storage, index, _binding, _first, _second = search_data
    index.set_results("consulta", [], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        with pytest.raises(QueryExecutionError, match="already open"):
            scan.open()
    finally:
        scan.close()

    assert index.closed is False


def test_explain_has_ranked_index_details(search_data) -> None:
    scan = _scan(search_data, request=3)
    plan = scan.explain()

    assert plan.op == "InvertedIndexScan"
    assert plan.detail == {
        "index": "idx_documentos_texto",
        "table": "documentos",
        "column": "contenido",
        "algorithm": "INVERTED",
        "requested_k": 3,
        "options": {"method": "bm25"},
    }
    assert plan.children == []
    assert plan.to_json()["op"] == "InvertedIndexScan"


def test_score_can_be_projected_as_float(search_data) -> None:
    _storage, index, _binding, first, _second = search_data
    index.set_results("consulta", [(first, 0.8)], method="bm25")
    scan = _scan(search_data)
    project = Project(
        scan,
        (
            SelectColumn(ColumnRef("id")),
            SelectColumn(BoundScoreExpr(2, "text"), alias="relevancia"),
        ),
    )

    assert project.schema == (
        ColumnDef("id", ColumnType.INT),
        ColumnDef("relevancia", ColumnType.FLOAT),
    )
    project.open()
    try:
        result = project.next()
        assert result is not None
        assert decode_row(result.data, project.schema) == (1, 0.8)
        assert project.next() is None
    finally:
        project.close()


def test_negative_finite_score_is_allowed(search_data) -> None:
    _storage, index, _binding, first, _second = search_data
    index.set_results("consulta", [(first, -0.1)], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        assert _drain(scan) == [(1, "primer documento", -0.1)]
    finally:
        scan.close()


def test_vector_scan_uses_the_same_rid_and_score_path(search_data) -> None:
    storage, _index, binding, first, _second = search_data
    vector_index = ScriptedRankedIndex("hnsw")
    vector_binding = replace(
        binding,
        name="idx_documentos_vector",
        algorithm="HNSW",
        index=vector_index,
        metric="cosine",
        dimension=2,
        profile_id="image-test",
    )
    vector_index.set_results((1.0, 0.0), [(first, 0.3)], metric="cosine", ef=8)
    scan = RankedScan(
        vector_binding,
        storage,
        (1.0, 0.0),
        SCHEMA,
        request=1,
        options={"metric": "cosine", "ef": 8},
        op_name="HNSWSearch",
    )

    scan.open()
    try:
        assert _drain(scan) == [(1, "primer documento", 0.3)]
        assert scan.explain().op == "HNSWSearch"
    finally:
        scan.close()

    assert vector_index.calls == [((1.0, 0.0), 1, {"metric": "cosine", "ef": 8})]
    assert vector_index.closed is False


@pytest.mark.parametrize("invalid_request", [-1, 1.5, True, None])
def test_invalid_request_rejected(search_data, invalid_request: object) -> None:
    storage, _index, binding, _first, _second = search_data
    with pytest.raises(QueryExecutionError, match="non-negative integer"):
        InvertedIndexScan(binding, storage, "consulta", SCHEMA, invalid_request, "bm25")


@pytest.mark.parametrize("method", ["TF_IDF", "BM_25", "cosine", ""])
def test_invalid_text_method_rejected(search_data, method: str) -> None:
    storage, _index, binding, _first, _second = search_data
    with pytest.raises(QueryExecutionError, match="tf_idf or bm25"):
        InvertedIndexScan(binding, storage, "consulta", SCHEMA, 2, method)


def test_text_scan_rejects_non_string_query(search_data) -> None:
    storage, _index, binding, _first, _second = search_data
    with pytest.raises(QueryExecutionError, match="string query"):
        InvertedIndexScan(binding, storage, (1, 2), SCHEMA, 2, "bm25")


def test_inverted_scan_rejects_vector_binding(search_data) -> None:
    storage, _index, binding, _first, _second = search_data
    vector_binding = replace(binding, algorithm="HNSW")
    with pytest.raises(QueryExecutionError, match="INVERTED binding"):
        InvertedIndexScan(vector_binding, storage, "consulta", SCHEMA, 2, "bm25")


def test_binding_column_must_match_storage_schema(search_data) -> None:
    storage, _index, binding, _first, _second = search_data
    invalid = replace(binding, column="desconocida")
    with pytest.raises(QueryExecutionError, match="unknown column"):
        InvertedIndexScan(invalid, storage, "consulta", SCHEMA, 2, "bm25")


@pytest.mark.parametrize("score", [None, 1, True, "0.8", nan, inf, -inf])
def test_invalid_provider_scores_are_rejected(search_data, score: object) -> None:
    _storage, index, _binding, first, _second = search_data
    index.set_results("consulta", [(first, score)], method="bm25")
    scan = _scan(search_data)

    with pytest.raises(QueryExecutionError, match="finite FLOAT"):
        scan.open()
    assert index.closed is False
    assert scan.explain().rows == 0


@pytest.mark.parametrize("rid", ["un_rid", RID(-1, 0), RID(0, -1), RID(True, 0)])
def test_invalid_provider_rids_are_rejected(search_data, rid: object) -> None:
    _storage, index, _binding, _first, _second = search_data
    index.set_results("consulta", [(rid, 0.8)], method="bm25")
    scan = _scan(search_data)

    with pytest.raises(QueryExecutionError, match="invalid RID"):
        scan.open()
    assert index.closed is False


@pytest.mark.parametrize("bad_hit", [1, (RID(0, 0),), [RID(0, 0), 0.4]])
def test_malformed_provider_hits_are_rejected(search_data, bad_hit: object) -> None:
    _storage, index, _binding, _first, _second = search_data
    index.set_results("consulta", [bad_hit], method="bm25")

    with pytest.raises(QueryExecutionError, match="invalid \\(RID, score\\) hit"):
        _scan(search_data).open()


def test_duplicate_hits_are_rejected(search_data) -> None:
    _storage, index, _binding, first, _second = search_data
    index.set_results("consulta", [(first, 0.9), (first, 0.8)], method="bm25")

    with pytest.raises(QueryExecutionError, match="duplicate RID"):
        _scan(search_data).open()


def test_ascending_scores_are_rejected(search_data) -> None:
    _storage, index, _binding, first, second = search_data
    index.set_results("consulta", [(first, 0.5), (second, 0.8)], method="bm25")

    with pytest.raises(QueryExecutionError, match="unsorted scores"):
        _scan(search_data).open()


def test_equal_scores_are_valid(search_data) -> None:
    _storage, index, _binding, first, second = search_data
    index.set_results("consulta", [(first, 0.5), (second, 0.5)], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        assert len(_drain(scan)) == 2
    finally:
        scan.close()


def test_provider_error_is_normalized_and_shared_index_remains_open(search_data) -> None:
    _storage, index, _binding, _first, _second = search_data
    index.set_error("consulta", ValueError("error interno"), method="bm25")
    scan = _scan(search_data)

    with pytest.raises(QueryExecutionError, match="ranked search failed"):
        scan.open()
    assert index.closed is False


def test_unknown_scripted_query_is_normalized(search_data) -> None:
    scan = _scan(search_data)

    with pytest.raises(QueryExecutionError, match="ranked search failed"):
        scan.open()


def test_stale_rid_raises_instead_of_silently_dropping_hit(search_data) -> None:
    storage, index, _binding, first, _second = search_data
    storage.remove(first)
    index.set_results("consulta", [(first, 0.9)], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        with pytest.raises(QueryExecutionError, match="stale RID"):
            scan.next()
    finally:
        scan.close()
    assert index.closed is False


def test_broken_record_is_a_query_error(search_data) -> None:
    storage, index, _binding, first, _second = search_data
    storage._records[first] = Record(data=b"broken")
    index.set_results("consulta", [(first, 0.9)], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        with pytest.raises(QueryExecutionError):
            scan.next()
    finally:
        scan.close()


def test_shared_lock_is_acquired_before_provider_is_called(search_data, monkeypatch) -> None:
    storage, index, _binding, _first, _second = search_data
    calls: list[str] = []
    original_top_k = index.top_k

    def lock_shared() -> None:
        calls.append("lock")

    def top_k(*args, **kwargs):
        calls.append("search")
        return original_top_k(*args, **kwargs)

    monkeypatch.setattr(storage, "lock_shared", lock_shared)
    monkeypatch.setattr(index, "top_k", top_k)
    index.set_results("consulta", [], method="bm25")
    scan = _scan(search_data)

    scan.open()
    try:
        assert calls == ["lock", "search"]
    finally:
        scan.close()
