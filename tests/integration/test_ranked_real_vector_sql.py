"""C22: SQL + persistent physical IVF/HNSW over real multimedia codebooks.

Descriptor extraction is mocked to make fixtures tiny and deterministic;
IVF/HNSW, page storage, ranking, SQL planning and Catalog persistence are real.
"""

import numpy as np
import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.indexes.vector.sql_factory import OnDiskVectorFactory
from engine.multimedia.codebook import save_codebook
from engine.multimedia.sql_profiles import MultimediaProfile
from engine.query import execute_sql
from engine.transactions.session import TransactionalSession

pytestmark = pytest.mark.integration


def _setup(tmp_path, monkeypatch, *, algorithm):
    from engine.multimedia import audio_features, image_features

    media = tmp_path / "media"
    media.mkdir()
    kind = "image" if algorithm == "HNSW" else "audio_segment"
    extension = "jpg" if kind == "image" else "wav"
    for key in ("a", "b", "c"):
        (media / f"{key}.{extension}").write_bytes(b"placeholder")

    dimension = 128 if kind == "image" else 20
    centroids = np.zeros((3, dimension), dtype=np.float32)
    centroids[1, 0] = 10
    centroids[2, 0] = 20
    codebook = tmp_path / "codebook.npz"
    save_codebook(codebook, centroids, np.ones(3))

    def descriptors_image(path):
        vector = np.zeros((1, 128), dtype=np.float32)
        vector[0, 0] = {"a": 0, "b": 10, "c": 20}[path.stem]
        return vector

    def audio_segments(path):
        yield 0, np.array([{"a": 0, "b": 10, "c": 20}[path.stem]])

    def descriptors_audio(signal):
        vector = np.zeros((1, 20), dtype=np.float32)
        vector[0, 0] = signal[0]
        return vector

    monkeypatch.setattr(image_features, "extract_sift", descriptors_image)
    monkeypatch.setattr(audio_features, "iter_audio_segments", audio_segments)
    monkeypatch.setattr(audio_features, "mfcc_from_signal", descriptors_audio)

    profile = MultimediaProfile("media-v1", kind, media, codebook)
    factory = OnDiskVectorFactory(
        tmp_path / "physical", {"media-v1": profile}, page_size=4096, nlist=3
    )
    db_path = tmp_path / "sql"
    catalog = Catalog(db_path, ranked_factory=factory)
    execute_sql("CREATE TABLE assets (id INT, path TEXT)", catalog)
    for ident, key in enumerate(("a", "b", "c"), start=1):
        execute_sql(f"INSERT INTO assets VALUES ({ident}, '{key}.{extension}')", catalog)
    return db_path, factory, catalog, codebook, extension


@pytest.mark.parametrize("algorithm", ["HNSW", "IVF"])
def test_physical_vector_index_sql_persists_and_drops(tmp_path, monkeypatch, algorithm):
    db_path, factory, catalog, _, ext = _setup(tmp_path, monkeypatch, algorithm=algorithm)
    metric = "cosine" if algorithm == "HNSW" else "euclidean"
    ddl = (
        f"CREATE INDEX idx_media ON assets (path) USING {algorithm} "
        f"WITH METRIC={metric}, DIMENSION=3, PROFILE='media-v1'"
    )
    execute_sql(ddl, catalog)
    binding = catalog.ranked_indexes_for("assets")["idx_media"]
    assert binding.mutable is True and binding.reversible is True
    assert binding.item_count == 3

    query = (
        "SELECT id, SIMILARITY_SCORE() AS score FROM assets "
        f"WHERE SIMILAR_TO('a.{ext}', k=3) USING {algorithm} "
        f"WITH METRIC={metric} ORDER BY score DESC"
    )
    first = execute_sql(query, catalog)
    assert first.rows[0][0] == 1
    assert len(first.rows) == 3
    assert all(type(row[1]) is float for row in first.rows)
    assert [row[1] for row in first.rows] == sorted([row[1] for row in first.rows], reverse=True)
    session = TransactionalSession(catalog)
    session.execute(f"INSERT INTO assets VALUES (4, 'a.{ext}')")
    after = execute_sql(query.replace("k=3", "k=4"), catalog)
    assert 4 in [row[0] for row in after.rows]
    session.execute("DELETE FROM assets WHERE id = 4")
    gone = execute_sql(query.replace("k=3", "k=4"), catalog)
    assert 4 not in [row[0] for row in gone.rows]
    assert len(execute_sql("SELECT * FROM assets", catalog).rows) == 3
    catalog.close()

    reopened = Catalog(db_path, ranked_factory=factory)
    try:
        assert execute_sql(query, reopened).rows == first.rows
        execute_sql("DROP INDEX idx_media", reopened)
        assert reopened.ranked_indexes_for("assets") == {}
        assert not factory._directory("idx_media").exists()
        execute_sql(ddl, reopened)
        assert len(execute_sql(query, reopened).rows) == 3
    finally:
        reopened.close()


def test_profile_change_is_rejected_when_reopening(tmp_path, monkeypatch):
    db_path, factory, catalog, codebook, _ = _setup(tmp_path, monkeypatch, algorithm="HNSW")
    execute_sql(
        "CREATE INDEX idx_media ON assets (path) USING HNSW "
        "WITH METRIC=cosine, DIMENSION=3, PROFILE='media-v1'",
        catalog,
    )
    catalog.close()
    centroids = np.zeros((3, 128), dtype=np.float32)
    centroids[0, 0] = 3
    save_codebook(codebook, centroids, np.ones(3))
    with pytest.raises(QueryExecutionError, match="profile/configuration changed"):
        Catalog(db_path, ranked_factory=factory)


def test_real_hnsw_index_returns_float_over_tcp(tmp_path, monkeypatch):
    import socket
    import threading

    from engine.transactions import protocol as proto
    from engine.transactions.connection_handler import ConnectionHandler
    from engine.transactions.transaction_manager import TransactionManager

    _, _, catalog, _, _ = _setup(tmp_path, monkeypatch, algorithm="HNSW")
    execute_sql(
        "CREATE INDEX ix ON assets (path) USING HNSW "
        "WITH METRIC=cosine, DIMENSION=3, PROFILE='media-v1'",
        catalog,
    )
    server = ConnectionHandler(
        catalog, TransactionManager(), host="127.0.0.1", port=0, max_connections=2
    )
    server.bind()
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        statement = (
            "SELECT id, SIMILARITY_SCORE() AS score FROM assets "
            "WHERE SIMILAR_TO('a.jpg', k=2) USING HNSW WITH METRIC=cosine "
            "ORDER BY score DESC"
        )
        with socket.create_connection(("127.0.0.1", server.port), timeout=3) as conn:
            conn.sendall(proto.encode_frame(proto.OP_QUERY, proto.encode_query(statement)))
            opcode, payload = proto.recv_frame(conn)
        assert opcode == proto.OP_RESULT, (
            proto.decode_error(payload) if opcode == proto.OP_ERROR else opcode
        )
        result = proto.decode_resultset(payload)
        assert result.rows[0][0] == 1
        assert result.columns[-1].type_name.value == "FLOAT"
        assert type(result.rows[0][-1]) is float
    finally:
        server.shutdown()
        catalog.close()
