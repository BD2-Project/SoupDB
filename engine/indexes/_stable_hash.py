"""Deterministic hashing for persistent index structures."""

import hashlib

from engine.indexes._key_codec import encode_key
from engine.indexes.base import Key

_HASH_BYTES = 8


def stable_hash(key: Key) -> int:
    """Return a deterministic unsigned 64-bit hash for an index key."""
    encoded = encode_key(key)
    digest = hashlib.blake2b(encoded, digest_size=_HASH_BYTES).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def directory_index(key: Key, global_depth: int) -> int:
    """Return the extendible-hash directory slot for a key."""
    if global_depth < 0 or global_depth > 64:
        raise ValueError("global_depth must be between 0 and 64")

    if global_depth == 0:
        return 0

    mask = (1 << global_depth) - 1
    return stable_hash(key) & mask
