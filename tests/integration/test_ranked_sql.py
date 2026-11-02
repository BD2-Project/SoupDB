"""C21: official ranked SQL scenarios through public API and actual TCP sockets.

The providers are scripted fakes: this verifies SQL/TCP integration, not real
TF-IDF/BM25 scoring or IVF/HNSW recall.
"""

import json
import socket
import threading

import pytest

from engine.common.ranked import RankedBinding
from engine.common.schema import ColumnDef, ColumnType
from engine.query import execute_sql
from engine.transactions import protocol as proto
from engine.transactions.connection_handler import ConnectionHandler
from engine.transactions.transaction_manager import TransactionManager
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_query_vectorizer import FakeQueryVectorizer
from tests.fakes.fake_ranked_search import ScriptedRankedIndex

pytestmark = pytest.mark.integration

SQL_TEXT_TFIDF = (
    "SELECT * FROM documentos WHERE MATCH(contenido, 'base de datos vectorial') "
    "USING TF_IDF LIMIT 10"
)
SQL_TEXT_BM25 = (
    "SELECT *, SCORE() AS relevancia FROM articulos "
    "WHERE MATCH(texto, 'machine learning') USING BM25 "
    "ORDER BY relevancia DESC"
)
SQL_IMAGE_HNSW = (
    "SELECT * FROM imagenes WHERE SIMILAR_TO('foto_consulta.jpg', k=2) "
    "USING HNSW WITH METRIC=cosine"
)
SQL_AUDIO_IVF = (
    "SELECT nombre, artista, SIMILARITY_SCORE() AS score FROM canciones "
    "WHERE SIMILAR_TO('audio_query.mp3', k=2) USING IVF WITH METRIC=euclidean "
    "ORDER BY score DESC"
)

SCENARIOS = (
    (SQL_TEXT_TFIDF, ("id", "contenido"), ((2, "segundo"), (1, "primero"))),
    (
        SQL_TEXT_BM25,
        ("id", "texto", "relevancia"),
        ((12, "aprendizaje b", 0.9), (11, "aprendizaje a", 0.4)),
    ),
    (
        SQL_IMAGE_HNSW,
        ("id", "archivo"),
        ((22, "imagen b.jpg"), (21, "imagen a.jpg")),
    ),
    (
        SQL_AUDIO_IVF,
        ("nombre", "artista", "score"),
        (("pista b", "beto", 0.8), ("pista a", "ana", -0.2)),
    ),
)


def _create_ranked(
    catalog: FakeCatalog,
    table: str,
    schema: tuple[ColumnDef, ...],
    rows: tuple[tuple[object, ...], ...],
    column: str,
    algorithm: str,
    query: str | tuple[float, ...],
    scores: tuple[tuple[int, float], ...],
    *,
    method: str | None = None,
    metric: str | None = None,
    profile: str | None = None,
) -> ScriptedRankedIndex:
    catalog.create_table(table, schema)
    for row in rows:
        catalog.insert(table, row)
    rids = [rid for rid, _ in catalog.file_org(table).scan()]
    kind = {"INVERTED": "text", "IVF": "ivf", "HNSW": "hnsw"}[algorithm]
    index = ScriptedRankedIndex(kind)
    options = {"method": method} if method is not None else {"metric": metric}
    index.set_results(query, [(rids[position], score) for position, score in scores], **options)
    catalog.add_ranked_index(
        RankedBinding(
            name=f"idx_{table}",
            table=table,
            column=column,
            algorithm=algorithm,
            index=index,
            metric=metric,
            dimension=len(query) if algorithm != "INVERTED" else None,
            profile_id=profile,
            item_count=len(rows),
            exact=True,
            exhaustive=True,
        )
    )
    return index


@pytest.fixture
def ranked_data():
    catalog = FakeCatalog()
    vectorizer = FakeQueryVectorizer()
    vectorizer.register("foto_consulta.jpg", "imagen-v1", (0.1, 0.2))
    vectorizer.register("audio_query.mp3", "audio-v1", (0.3, 0.4))
    catalog.query_vectorizer = vectorizer
    indexes = {}

    indexes["tfidf"] = _create_ranked(
        catalog,
        "documentos",
        (ColumnDef("id", ColumnType.INT), ColumnDef("contenido", ColumnType.TEXT)),
        ((1, "primero"), (2, "segundo")),
        "contenido",
        "INVERTED",
        "base de datos vectorial",
        ((1, 0.8), (0, 0.5)),
        method="tf_idf",
    )
    indexes["bm25"] = _create_ranked(
        catalog,
        "articulos",
        (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT)),
        ((11, "aprendizaje a"), (12, "aprendizaje b")),
        "texto",
        "INVERTED",
        "machine learning",
        ((1, 0.9), (0, 0.4)),
        method="bm25",
    )
    indexes["hnsw"] = _create_ranked(
        catalog,
        "imagenes",
        (ColumnDef("id", ColumnType.INT), ColumnDef("archivo", ColumnType.TEXT)),
        ((21, "imagen a.jpg"), (22, "imagen b.jpg")),
        "archivo",
        "HNSW",
        (0.1, 0.2),
        ((1, 0.95), (0, 0.6)),
        metric="cosine",
        profile="imagen-v1",
    )
    indexes["ivf"] = _create_ranked(
        catalog,
        "canciones",
        (ColumnDef("nombre", ColumnType.TEXT), ColumnDef("artista", ColumnType.TEXT)),
        (("pista a", "ana"), ("pista b", "beto")),
        "nombre",
        "IVF",
        (0.3, 0.4),
        ((1, 0.8), (0, -0.2)),
        metric="euclidean",
        profile="audio-v1",
    )
    return catalog, vectorizer, indexes


@pytest.fixture
def tcp_server(ranked_data):
    catalog, _, _ = ranked_data
    handler = ConnectionHandler(
        catalog,
        TransactionManager(),
        host="127.0.0.1",
        port=0,
        max_connections=4,
        socket_timeout=3,
        lock_timeout_ms=1000,
    )
    handler.bind()
    worker = threading.Thread(target=handler.serve_forever, daemon=True)
    worker.start()
    try:
        yield handler
    finally:
        handler.shutdown()


def _request(connection: socket.socket, sql: str):
    connection.sendall(proto.encode_frame(proto.OP_QUERY, proto.encode_query(sql)))
    return proto.recv_frame(connection)


@pytest.mark.parametrize(("sql", "names", "rows"), SCENARIOS)
def test_official_queries_via_public_api(ranked_data, sql, names, rows) -> None:
    catalog, _, _ = ranked_data
    result = execute_sql(sql, catalog)
    assert tuple(column.name for column in result.columns) == names
    assert result.rows == rows
    assert all("\x00" not in column.name for column in result.columns)


@pytest.mark.parametrize(("sql", "names", "rows"), SCENARIOS)
def test_official_queries_over_real_tcp(tcp_server, sql, names, rows) -> None:
    with socket.create_connection(("127.0.0.1", tcp_server.port), timeout=3) as conn:
        opcode, payload = _request(conn, sql)
    if opcode == proto.OP_ERROR:
        pytest.fail(f"TCP rejected ranked SQL: {proto.decode_error(payload)}")
    assert opcode == proto.OP_RESULT
    result = proto.decode_resultset(payload)
    assert tuple(column.name for column in result.columns) == names
    assert result.rows == rows
    assert all("\x00" not in column.name for column in result.columns)
    if "relevancia" in names or "score" in names:
        assert result.columns[-1].type_name is ColumnType.FLOAT
        assert all(type(row[-1]) is float for row in result.rows)


@pytest.mark.parametrize(
    ("sql", "expected_node"),
    (
        (SQL_TEXT_TFIDF, "InvertedIndexScan"),
        (SQL_TEXT_BM25, "InvertedIndexScan"),
        (SQL_IMAGE_HNSW, "HNSWSearch"),
        (SQL_AUDIO_IVF, "IVFSearch"),
    ),
)
def test_explain_json_does_not_execute_search(ranked_data, sql, expected_node) -> None:
    catalog, vectorizer, indexes = ranked_data
    result = execute_sql("EXPLAIN (FORMAT JSON) " + sql, catalog)
    plan = json.loads(result.rows[0][0])

    def walk(node):
        yield node
        for child in node["children"]:
            yield from walk(child)

    leaves = [node for node in walk(plan) if node["op"] == expected_node]
    assert len(leaves) == 1
    assert not any(index.calls for index in indexes.values())
    assert vectorizer.calls == []


def test_tcp_worker_survives_ranked_execution_error(tcp_server) -> None:
    with socket.create_connection(("127.0.0.1", tcp_server.port), timeout=3) as conn:
        opcode, payload = _request(
            conn,
            "SELECT * FROM documentos WHERE MATCH(contenido, 'sin script') USING BM25 LIMIT 1",
        )
        assert opcode == proto.OP_ERROR
        code, message = proto.decode_error(payload)
        assert code == proto.ERR_EXECUTION
        assert "ranked" in message.lower()

        opcode, payload = _request(conn, SQL_TEXT_TFIDF)
        assert opcode == proto.OP_RESULT
        assert proto.decode_resultset(payload).rows[0][0] == 2
