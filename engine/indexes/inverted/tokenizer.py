"""Deterministic ES/EN tokenization without external language resources.

NFKC and casefold normalize width/case; accents and ñ remain significant.
Words and numbers are retained; punctuation and underscores split tokens.
Stopwords are removed before stemming. The stemmer is a bilingual heuristic,
not Porter or a language detector; its rules and configuration are versioned.
"""

import re
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass

TOKENIZER_VERSION = 1
DEFAULT_STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for from with as is are was were "
    "be been being it its this that these those i you he she we they me my your "
    "our their de del la las el los y o en por para con un una unos unas es son "
    "ser se que como al lo le les su sus".split()
)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_VOWELS = frozenset("aeiouáéíóúü")
_UNCHANGED = frozenset({"series", "species", "caries", "analysis", "news", "virus"})


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def stem(token: str) -> str:
    """Reduce inflections without accent folding; numbers stay unchanged.

    Plurals: iones→ión, ces→z, ies→y, selected consonant+es→consonant,
    final s except us/is/ss. Then at most one adverb/verb rule: mente/ness/ly
    stripping with >=4 letters left, or ing/ed with >=3 letters and a vowel.
    Doubled final consonants other than l/s/z collapse after ing/ed removal.
    Silent e is not reconstructed. Irregulars and language disambiguation are
    outside this heuristic, which can conflate words across the two languages.
    """
    if len(token) < 4 or not token.isalpha() or token in _UNCHANGED:
        return token
    if len(token) >= 7 and token.endswith("iones"):
        token = token[:-5] + "ión"
    elif len(token) >= 5 and token.endswith("ces"):
        token = token[:-3] + "z"
    elif len(token) >= 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) >= 6 and token.endswith("es") and token[:-2].endswith(
        ("l", "r", "n", "d", "x", "z", "ch", "sh", "ss")
    ):
        token = token[:-2]
    elif len(token) >= 5 and token.endswith("s") and not token.endswith(("us", "is", "ss")):
        token = token[:-1]

    for suffix, minimum in (("mente", 4), ("ness", 4), ("ly", 4)):
        if token.endswith(suffix) and len(token) - len(suffix) >= minimum:
            return token[: -len(suffix)]
    for suffix in ("ing", "ed"):
        if not token.endswith(suffix):
            continue
        base = token[: -len(suffix)]
        if len(base) >= 3 and any(char in _VOWELS for char in base):
            if (
                len(base) >= 4
                and base[-1] == base[-2]
                and base[-1] not in _VOWELS | {"l", "s", "z"}
            ):
                base = base[:-1]
            return base
    return token


@dataclass(frozen=True)
class Tokenizer:
    """Immutable configuration shared by document and query processing.

    An empty stoplist disables filtering; custom stopwords replace defaults.
    """

    stopwords: frozenset[str] = DEFAULT_STOPWORDS
    stemming: bool = True

    def __post_init__(self) -> None:
        if type(self.stemming) is not bool:
            raise TypeError("stemming must be a boolean")
        if isinstance(self.stopwords, str):
            raise TypeError("stopwords must be a collection of words")
        normalized: set[str] = set()
        for word in self.stopwords:
            if not isinstance(word, str):
                raise TypeError("stopwords must contain strings")
            word = _normalize(word).strip()
            if not _WORD.fullmatch(word):
                raise ValueError("each stopword must be a single non-empty word")
            normalized.add(word)
        object.__setattr__(self, "stopwords", frozenset(normalized))

    def __call__(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        words = (word for word in _WORD.findall(_normalize(text)) if word not in self.stopwords)
        return [stem(word) if self.stemming else word for word in words]


DEFAULT_TOKENIZER = Tokenizer()


def tokenize(
    text: str,
    *,
    stopwords: Collection[str] | None = None,
    stemming: bool = True,
) -> list[str]:
    """Tokenize with bilingual defaults or an explicit replacement stoplist."""
    if isinstance(stopwords, str):
        raise TypeError("stopwords must be a collection of words")
    return Tokenizer(
        stopwords=DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords),
        stemming=stemming,
    )(text)
