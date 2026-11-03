from dataclasses import FrozenInstanceError

import pytest

from engine.indexes.inverted.tokenizer import Tokenizer, stem, tokenize


def test_unicode_normalization_case_punctuation_and_numbers() -> None:
    assert tokenize("  ¡ÁRBOL, NIÑO! cafe\u0301; ＤＢ_2026  ", stemming=False) == [
        "árbol", "niño", "café", "db", "2026"
    ]


def test_bilingual_stopwords_preserve_content_words() -> None:
    assert tokenize("El índice y la consulta with the database", stemming=False) == [
        "índice", "consulta", "database"
    ]


def test_custom_stopwords_replace_defaults_and_normalize() -> None:
    assert tokenize("THE árbol consulta", stopwords={" ÁRBOL "}, stemming=False) == [
        "the", "consulta"
    ]
    assert tokenize("El and", stopwords=(), stemming=False) == ["el", "and"]


@pytest.mark.parametrize("text", ["", " \t\n", "!!!___", "the y con"])
def test_empty_and_stopword_only_input(text: str) -> None:
    assert tokenize(text) == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("acciones", "acción"), ("luces", "luz"), ("árboles", "árbol"),
        ("libros", "libro"), ("queries", "query"), ("indexes", "index"),
        ("databases", "database"), ("running", "run"), ("indexed", "index"),
        ("quickly", "quick"), ("happiness", "happi"), ("claramente", "clara"),
        ("series", "series"), ("virus", "virus"), ("analysis", "analysis"),
        ("is", "is"), ("2026", "2026"), ("tree", "tree"),
    ],
)
def test_documented_stemming_rules(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_stemming_can_be_disabled() -> None:
    assert tokenize("RUNNING acciones", stemming=False) == ["running", "acciones"]


def test_configuration_is_immutable_and_processing_deterministic() -> None:
    tokenizer = Tokenizer(stopwords=frozenset({"THE"}))
    assert tokenizer("The running libros") == tokenizer("THE RUNNING LIBROS") == ["run", "libro"]
    assert tokenizer.stopwords == frozenset({"the"})
    with pytest.raises(FrozenInstanceError):
        tokenizer.stemming = False


@pytest.mark.parametrize("stopwords", ["the", {1}])
def test_invalid_stopword_types_are_rejected(stopwords) -> None:
    with pytest.raises(TypeError):
        tokenize("text", stopwords=stopwords)


@pytest.mark.parametrize("stopwords", [{""}, {"two words"}, {"x_y"}])
def test_stopwords_must_be_single_words(stopwords) -> None:
    with pytest.raises(ValueError):
        tokenize("text", stopwords=stopwords)


def test_invalid_text_and_stemming_types_are_rejected() -> None:
    with pytest.raises(TypeError):
        tokenize(None)
    with pytest.raises(TypeError):
        tokenize("text", stemming=1)
