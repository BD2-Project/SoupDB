"""Compatibility shim: vector-index code moved to the ``soupdb-vector`` plugin.

Importing this package requires the plugin to be installed. The submodules are
aliased into this namespace so existing ``engine.indexes.vector.*`` imports
keep working during the migration window.
"""

import sys as _sys

from soupdb_vector.indexes import (
    _record_file,
    brute_force,
    constants,
    hnsw,
    ivf,
    kmeans,
    metrics,
    sql_factory,
    store,
)
from soupdb_vector.indexes.brute_force import BruteForceIndex
from soupdb_vector.indexes.hnsw import HNSWIndex
from soupdb_vector.indexes.ivf import IVFIndex
from soupdb_vector.indexes.metrics import COSINE, EUCLIDEAN, INNER_PRODUCT, METRICS
from soupdb_vector.indexes.store import VectorStore

_SUBMODULES = (
    _record_file,
    brute_force,
    constants,
    hnsw,
    ivf,
    kmeans,
    metrics,
    sql_factory,
    store,
)

for _module in _SUBMODULES:
    _sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module

__all__ = [
    "COSINE",
    "EUCLIDEAN",
    "INNER_PRODUCT",
    "METRICS",
    "BruteForceIndex",
    "HNSWIndex",
    "IVFIndex",
    "VectorStore",
]
