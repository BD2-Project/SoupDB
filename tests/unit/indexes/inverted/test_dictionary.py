from pathlib import Path

import pytest

from engine.indexes.inverted.dictionary import DictionaryEntry, TermDictionary


def test_hash_lookup_sidecar_iteration_and_reopen(tmp_path: Path) -> None:
    dictionary = TermDictionary.create(tmp_path, page_size=128)
    entries = [(f"term-{i}", DictionaryEntry(i + 1, 8 + i * 100, 8 + (i + 1) * 16))
               for i in range(160)]
    for term, entry in entries:
        dictionary.add(term, entry)
    dictionary.close()
    dictionary.close()
    reopened = TermDictionary.load(tmp_path, page_size=128)
    assert list(reopened.items()) == entries
    for term, entry in reopened.items():
        assert reopened.find(term) == entry  # Interleaved seeks must not disturb iteration.
    assert reopened.find("unknown") is None
    with pytest.raises(ValueError):
        reopened.add("new", DictionaryEntry(1, 8, 24))
    reopened.close()


def test_invalid_terms_entries_and_duplicates(tmp_path: Path) -> None:
    dictionary = TermDictionary.create(tmp_path, page_size=128)
    entry = DictionaryEntry(1, 8, 24)
    for term, value in [("", entry), ("ñ" * 56, entry),
                        ("bad", DictionaryEntry(0, 8, 8)),
                        ("bad", DictionaryEntry(1, 8, 999))]:
        with pytest.raises(ValueError):
            dictionary.add(term, value)
    dictionary.add("term", entry)
    with pytest.raises(ValueError):
        dictionary.add("term", entry)
    dictionary.close()


def test_missing_dictionary_does_not_create_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TermDictionary.load(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("corruption", ["header", "truncated", "hash", "reference"])
def test_corruption_rejected_and_resources_released(tmp_path: Path, corruption: str) -> None:
    dictionary = TermDictionary.create(tmp_path)
    dictionary.add("term", DictionaryEntry(1, 8, 24))
    dictionary.close()
    path = tmp_path / ("terms.hash" if corruption == "hash" else "terms.meta")
    data = bytearray(path.read_bytes())
    if corruption == "header" or corruption == "hash":
        data[0] ^= 1
    elif corruption == "reference":
        data[-1] ^= 1
    else:
        data = data[:-1]
    path.write_bytes(data)
    with pytest.raises(ValueError):
        TermDictionary.load(tmp_path)
    # On Windows this also checks handles were closed on failed load.
    for file in tmp_path.iterdir():
        file.unlink()
