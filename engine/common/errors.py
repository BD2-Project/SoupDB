"""Own exceptions for the engine.

Nothing in the engine raises a bare ``Exception``.
"""


class SoupDBError(Exception):
    """Base class for all engine exceptions."""


class UnsupportedOperation(SoupDBError, NotImplementedError):
    """Operation not supported by a given implementation.

    Raised, for example, by ``range_search`` on hash-based indexes.
    """


class ContractViolation(SoupDBError):
    """An implementation broke a frozen contract."""


class RecordNotFound(SoupDBError):
    """The requested record does not exist."""


class TransactionError(SoupDBError):
    """Invalid transaction state or concurrency control failure."""


class QueryParseError(SoupDBError):
    """Invalid SQL syntax found while tokenizing or parsing."""


class QueryExecutionError(SoupDBError):
    """Invalid values found while evaluating an executed query."""
