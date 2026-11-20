"""Record identifier: unique reference to a record on disk."""

from typing import NamedTuple


class RID(NamedTuple):
    """Identifies a record by its page and slot.

    page_id: identifier of the page on disk.
    slot: position of the record within the page.
    """

    page_id: int
    slot: int
