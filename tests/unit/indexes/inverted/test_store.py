import json

import pytest

from engine.common.rid import RID
from engine.indexes.inverted.store import InvertedStore
from engine.indexes.inverted.tokenizer import Tokenizer

TOKENIZER = Tokenizer(stopwords=frozenset({"omit"}), stemming=False)
ITEMS = [("Árbol disk disk omit", RID(7, 2)), ("disk page árbol", RID(2, 4)), ("", RID(9, 0))]


def test_store_reopens_complete_statistics_postings_and_analyzer(tmp_path):
    path = tmp_path / "index"
    store = InvertedStore.build(path, ITEMS, tokenizer=TOKENIZER, memory_limit_bytes=1, fan_in=2)
    try:
        assert store.N == 3
        assert store.avgdl == 2.0
        assert store.doc_lengths == {0: 3, 1: 3, 2: 0}
        assert store.rids == {0: RID(7, 2), 1: RID(2, 4), 2: RID(9, 0)}
        assert store.tokenizer == TOKENIZER
        assert store.postings("disk") == [(0, 2), (1, 1)]
        assert store.postings("árbol") == [(0, 1), (1, 1)]
        assert store.postings("missing") == []
        terms = [(term, entry.df) for term, entry in store.terms()]
        assert terms == [("disk", 2), ("page", 1), ("árbol", 2)]
        # Term enumeration remains valid when lookups seek the same sidecar.
        assert [(term, store.postings(term)) for term, _ in store.terms()] == [
            ("disk", [(0, 2), (1, 1)]), ("page", [(1, 1)]), ("árbol", [(0, 1), (1, 1)]),
        ]
    finally:
        store.close()
    store.close()
    reopened = InvertedStore.load(path)
    try:
        assert reopened.tokenizer == TOKENIZER
        assert reopened.N == 3 and reopened.avgdl == 2.0
        assert reopened.postings("disk") == [(0, 2), (1, 1)]
    finally:
        reopened.close()


@pytest.mark.parametrize("items", [[], [("the and", RID(3, 0)), ("", RID(4, 0))]])
def test_zero_length_corpus_roundtrip(tmp_path, items):
    path = tmp_path / "empty"
    store = InvertedStore.build(path, items)
    store.close()
    store = InvertedStore.load(path)
    try:
        assert store.N == len(items)
        assert store.avgdl == 0.0
        assert list(store.terms()) == []
        assert store.postings("unknown") == []
    finally:
        store.close()


def test_failed_build_cleans_staging_and_preserves_other_files(tmp_path):
    sentinel = tmp_path / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    path = tmp_path / "failed"
    with pytest.raises(ValueError, match="duplicate"):
        InvertedStore.build(path, [("alpha", RID(1, 0)), ("beta", RID(1, 0))],
                            memory_limit_bytes=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep"]


def test_build_never_overwrites_existing_directory(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    sentinel = path / "keep"
    sentinel.write_bytes(b"original")
    with pytest.raises((FileExistsError, ValueError)):
        InvertedStore.build(path, ITEMS)
    assert sentinel.read_bytes() == b"original"
    assert list(path.iterdir()) == [sentinel]


def test_load_missing_artifacts_does_not_create_them(tmp_path):
    path = tmp_path / "absent"
    with pytest.raises((OSError, ValueError)):
        InvertedStore.load(path)
    assert not path.exists()
    path.mkdir()
    with pytest.raises((OSError, ValueError)):
        InvertedStore.load(path)
    assert list(path.iterdir()) == []


def test_store_rejects_reads_after_close(tmp_path):
    store = InvertedStore.build(tmp_path / "index", ITEMS)
    store.close()
    with pytest.raises((ValueError, RuntimeError)):
        store.postings("disk")


@pytest.mark.parametrize("field,value", [("version", 999), ("N", 200), ("avgdl", -1.0)])
def test_manifest_inconsistency_is_rejected(tmp_path, field, value):
    path = tmp_path / "index"
    InvertedStore.build(path, ITEMS).close()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        InvertedStore.load(path)


def test_missing_hash_is_rejected_without_replacement(tmp_path):
    path = tmp_path / "index"
    InvertedStore.build(path, ITEMS).close()
    hash_path = path / "terms.hash"
    hash_path.unlink()
    with pytest.raises((OSError, ValueError)):
        InvertedStore.load(path)
    assert not hash_path.exists()
