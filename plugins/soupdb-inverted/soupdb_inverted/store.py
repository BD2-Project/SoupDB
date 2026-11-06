"""Durable SPIMI artifacts, published only after a complete reopen validation."""

import json
import math
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing
from pathlib import Path

from engine.common.rid import RID
from soupdb_inverted.dictionary import DictionaryEntry, TermDictionary
from soupdb_inverted.merge import merged_terms
from soupdb_inverted.postings import PostingsFile
from soupdb_inverted.spimi import spimi_runs
from soupdb_inverted.tokenizer import DEFAULT_TOKENIZER, TOKENIZER_VERSION, Tokenizer

_VERSION = 1
_ARTIFACTS = ("manifest.json", "postings.bin", "terms.meta", "terms.hash")


class InvertedStore:
    """Immutable dense-document postings with a persistent ExtendibleHash dictionary."""

    def __init__(
        self,
        dictionary: TermDictionary,
        postings: PostingsFile,
        tokenizer: Tokenizer,
        rids: dict[int, RID],
        lengths: dict[int, int],
    ) -> None:
        self._dictionary = dictionary
        self._postings = postings
        self.tokenizer = tokenizer
        self.rids = rids
        self.doc_lengths = lengths
        self.N = len(rids)
        self.avgdl = sum(lengths.values()) / self.N if self.N else 0.0
        self._closed = False

    @classmethod
    def build(
        cls,
        path: str | Path,
        items: Iterable[tuple[str, RID]],
        *,
        tokenizer: Tokenizer = DEFAULT_TOKENIZER,
        memory_limit_bytes: int = 8 * 1024 * 1024,
        page_size: int = 4096,
        fan_in: int = 8,
    ) -> "InvertedStore":
        """Stage, validate, and rename into a new directory; never replace an index.

        Temporary blocks and failed stages are owned by this call. Publication
        is a same-filesystem directory rename after all artifacts are closed.
        """
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"index already exists: {target}")
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("tokenizer must be a Tokenizer configuration")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".inverted-", dir=target.parent) as temporary:
            working = Path(temporary)
            stage = working / "index"
            stage.mkdir()
            built = spimi_runs(
                items, working / "runs", tokenizer=tokenizer, memory_limit_bytes=memory_limit_bytes
            )
            with ExitStack() as stack:
                dictionary = TermDictionary.create(stage, page_size=page_size)
                stack.callback(dictionary.close)
                postings = PostingsFile.create(stage / "postings.bin")
                stack.callback(postings.close)
                with closing(merged_terms(built.runs, fan_in=fan_in, temp_dir=working)) as terms:
                    for term, entries in terms:
                        offset, length = postings.append(entries)
                        dictionary.add(term, DictionaryEntry(len(entries), offset, length))
            count = len(built.rids)
            manifest = {
                "version": _VERSION,
                "page_size": page_size,
                "tokenizer": {
                    "version": TOKENIZER_VERSION,
                    "stopwords": sorted(tokenizer.stopwords),
                    "stemming": tokenizer.stemming,
                },
                "documents": [
                    [rid.page_id, rid.slot, built.doc_lengths[doc_id]]
                    for doc_id, rid in built.rids.items()
                ],
                "N": count,
                "avgdl": sum(built.doc_lengths.values()) / count if count else 0.0,
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
            checked = cls.load(stage)
            checked.close()
            stage.rename(target)
        return cls.load(target)

    @classmethod
    def load(cls, path: str | Path) -> "InvertedStore":
        """Reopen existing artifacts without corpus access or creating missing files."""
        root = Path(path)
        if any(not (root / name).is_file() for name in _ARTIFACTS):
            raise FileNotFoundError("inverted index artifacts are missing")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("version")) is not int
            or manifest.get("version") != _VERSION
        ):
            raise ValueError("invalid inverted store version")
        page_size = manifest.get("page_size")
        if type(page_size) is not int or page_size < 128:
            raise ValueError("invalid inverted store page size")
        config = manifest.get("tokenizer")
        if (
            not isinstance(config, dict)
            or type(config.get("version")) is not int
            or config.get("version") != TOKENIZER_VERSION
            or not isinstance(config.get("stopwords"), list)
            or type(config.get("stemming")) is not bool
        ):
            raise ValueError("invalid persisted tokenizer configuration")
        try:
            tokenizer = Tokenizer(frozenset(config["stopwords"]), config["stemming"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid persisted stopwords") from exc
        documents = manifest.get("documents")
        count = manifest.get("N")
        if (
            not isinstance(documents, list)
            or type(count) is not int
            or count < 0
            or count != len(documents)
        ):
            raise ValueError("inconsistent document count")
        rids: dict[int, RID] = {}
        lengths: dict[int, int] = {}
        for doc_id, document in enumerate(documents):
            if (
                not isinstance(document, list)
                or len(document) != 3
                or any(type(value) is not int or value < 0 for value in document)
            ):
                raise ValueError("invalid persisted document metadata")
            rids[doc_id] = RID(document[0], document[1])
            lengths[doc_id] = document[2]
        if len(set(rids.values())) != count:
            raise ValueError("duplicate persisted document RID")
        expected_avg = sum(lengths.values()) / count if count else 0.0
        avg = manifest.get("avgdl")
        if (
            type(avg) is not float
            or not math.isfinite(avg)
            or not math.isclose(avg, expected_avg, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("inconsistent average document length")
        dictionary = TermDictionary.load(root, page_size=page_size)
        try:
            postings = PostingsFile.load(root / "postings.bin")
        except BaseException:
            dictionary.close()
            raise
        store = cls(dictionary, postings, tokenizer, rids, lengths)
        try:
            observed = dict.fromkeys(rids, 0)
            previous = ""
            for term, entry in dictionary.items():
                if term <= previous or entry.df > count:
                    raise ValueError("invalid term ordering or document frequency")
                previous = term
                for doc_id, tf in postings.read(entry.offset, entry.length, df=entry.df):
                    if doc_id not in observed:
                        raise ValueError("posting references an unknown document")
                    observed[doc_id] += tf
            if observed != lengths:
                raise ValueError("postings frequencies differ from document lengths")
            return store
        except BaseException:
            store.close()
            raise

    def _check(self) -> None:
        if self._closed:
            raise ValueError("inverted store is closed")

    def terms(self) -> Iterator[tuple[str, DictionaryEntry]]:
        self._check()
        yield from self._dictionary.items()

    def postings(self, term: str) -> list[tuple[int, int]]:
        self._check()
        entry = self._dictionary.find(term)
        if entry is None:
            return []
        return self._postings.read(entry.offset, entry.length, df=entry.df)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._dictionary.close()
        finally:
            self._postings.close()
