"""Compatibility shim: inverted-index code moved to the ``soupdb-inverted`` plugin.

Importing this package requires the plugin to be installed. The submodules are
aliased into this namespace so existing ``engine.indexes.inverted.*`` imports
keep working during the migration window.
"""

import sys as _sys

from soupdb_inverted import (
    corpus_reader,
    dictionary,
    index,
    merge,
    postings,
    provider,
    scoring,
    spimi,
    store,
    tokenizer,
    utils,
)

_SUBMODULES = (
    corpus_reader,
    dictionary,
    index,
    merge,
    postings,
    provider,
    scoring,
    spimi,
    store,
    tokenizer,
    utils,
)

for _module in _SUBMODULES:
    _sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module

__all__ = [module.__name__.rsplit(".", 1)[-1] for module in _SUBMODULES]
