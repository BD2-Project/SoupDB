import json

import pytest

from engine.indexes.inverted.corpus_reader import read_corpus


def _corpus(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    serialized = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(serialized, encoding="utf-8")
    return path


def test_reads_confirmed_corpus_schema_and_preserves_unicode(tmp_path):
    path = _corpus(tmp_path, [
        {"_id": "paper-1", "title": "Índices", "text": "Búsqueda en español."},
        {"_id": "paper-2", "title": "", "text": "English retrieval."},
    ])
    docs = list(read_corpus(path))
    assert [(doc.source_id, doc.text) for doc in docs] == [
        ("paper-1", "Índices Búsqueda en español."),
        ("paper-2", "English retrieval."),
    ]


@pytest.mark.parametrize("id_field", ["_id", "id", "doc_id"])
@pytest.mark.parametrize("text_field", ["text", "contenido", "body", "content"])
def test_supported_aliases_and_integer_ids(tmp_path, id_field, text_field):
    docs = list(read_corpus(_corpus(tmp_path, [{id_field: 17, text_field: "hello"}])))
    assert [(doc.source_id, doc.text) for doc in docs] == [("17", "hello")]


def test_empty_document_is_preserved(tmp_path):
    docs = list(read_corpus(_corpus(tmp_path, [{"_id": "empty", "text": ""}])))
    assert docs[0].text == ""


def test_bad_lines_are_logged_and_valid_lines_continue(tmp_path, caplog):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        'invalid json\n[]\n{"_id":"missing"}\n'
        '{"_id":"bad-text","text":42}\n'
        '{"_id":true,"text":"invalid id"}\n'
        '{"_id":"ok","text":"valid"}\n', encoding="utf-8",
    )
    docs = list(read_corpus(path))
    assert [(doc.source_id, doc.text) for doc in docs] == [("ok", "valid")]
    assert len(caplog.records) >= 5
    assert str(path) in caplog.text


def test_duplicate_ids_are_rejected_after_normalization(tmp_path):
    path = _corpus(tmp_path, [{"id": 1, "text": "first"}, {"id": "1", "text": "second"}])
    with pytest.raises(ValueError, match="duplicate"):
        list(read_corpus(path))


def test_missing_corpus_does_not_create_file(tmp_path):
    path = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        list(read_corpus(path))
    assert not path.exists()
