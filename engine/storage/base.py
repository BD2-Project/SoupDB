"""Frozen contract for every file organization.

All file organizations return RIDs. Every implementation of
:class:`FileOrganization` must pass the conformance suite.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from engine.common.record import Record
from engine.common.rid import RID


class FileOrganization(ABC):
    """Base interface for all file organizations."""

    @abstractmethod
    def insert(self, record: Record) -> RID:
        """Insert a record and return its RID."""

    @abstractmethod
    def fetch(self, rid: RID) -> Record | None:
        """Fetch the record at a RID, or None if it does not exist."""

    @abstractmethod
    def remove(self, rid: RID) -> bool:
        """Remove the record at a RID; returns whether it was removed."""

    @abstractmethod
    def scan(self) -> Iterator[tuple[RID, Record]]:
        """Yield every (RID, record) pair currently stored."""

    def lock_shared(self) -> None:  # noqa: B027  (un almacen sin bloqueos no tiene nada que tomar)
        """Take a shared lock on the whole table before reading any of it.

        ``scan`` takes that lock by itself, and ``insert``/``remove`` take it in
        exclusive mode, but a reader that walks rows one RID at a time through
        ``fetch`` would otherwise never lock the table and could read a row another
        transaction has written but not committed. Readers using an index to find
        the RIDs and ``fetch`` to read them call this first, so that reading by
        index and reading by table obey the same locking protocol.

        A file organization with no locking layer has nothing to take, so the
        default does nothing.
        """
