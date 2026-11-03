"""Read JSONL documents, logging malformed lines and rejecting duplicate IDs.

The seen-ID set grows with corpus size. Source IDs are strings and are not
storage RIDs. Blank documents remain valid, so N includes empty documents.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    source_id: str
    text: str


def read_corpus(path: str | Path) -> Iterator[CorpusDocument]:
    """Read _id/id/doc_id and text/contenido/body/content plus optional title.

    Invalid JSON, objects or fields are skipped with path and line diagnostics.
    IDs must be nonblank strings or integers (not booleans); aliases use the
    order above. Duplicate normalized IDs fail instead of silently replacing.
    """
    path = Path(path)
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("document must be a JSON object")
                source_id = next((row[k] for k in ("_id", "id", "doc_id") if k in row), None)
                if type(source_id) not in (str, int) or not str(source_id).strip():
                    raise ValueError("document ID must be a nonblank string or integer")
                source_id = str(source_id).strip()
                text = next(
                    (row[k] for k in ("text", "contenido", "body", "content") if k in row),
                    None,
                )
                title = row.get("title", "")
                if not isinstance(text, str) or not isinstance(title, str):
                    raise ValueError("text and optional title must be strings")
            except (ValueError, TypeError) as exc:
                _LOGGER.warning("Skipping malformed document %s:%d: %s", path, line_number, exc)
                continue
            if source_id in seen:
                raise ValueError(f"duplicate document ID {source_id!r} at {path}:{line_number}")
            seen.add(source_id)
            yield CorpusDocument(source_id, " ".join(part for part in (title, text) if part))
