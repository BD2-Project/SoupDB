"""Compatibility shim: multimedia code moved to the ``soupdb-vector`` plugin.

Importing this package requires the plugin to be installed. The submodules are
aliased into this namespace so existing ``engine.multimedia.*`` imports keep
working during the migration window.
"""

import sys as _sys

from soupdb_vector.multimedia import (
    audio_features,
    build_vectors,
    codebook,
    descriptor_cache,
    image_features,
    manifest,
    sql_profiles,
    vector_file,
)

_SUBMODULES = (
    audio_features,
    build_vectors,
    codebook,
    descriptor_cache,
    image_features,
    manifest,
    sql_profiles,
    vector_file,
)

for _module in _SUBMODULES:
    _sys.modules[f"{__name__}.{_module.__name__.rsplit('.', 1)[-1]}"] = _module

__all__ = [module.__name__.rsplit(".", 1)[-1] for module in _SUBMODULES]
