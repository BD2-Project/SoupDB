"""Stable binary codec for index keys."""

import struct
from typing import Any

_INT_TAG = b"i"
_STR_TAG = b"s"
_TUPLE_TAG = b"t"

_INT_FORMAT = "<q"
_LENGTH_FORMAT = "<I"

_INT_SIZE = struct.calcsize(_INT_FORMAT)
_LENGTH_SIZE = struct.calcsize(_LENGTH_FORMAT)


def encode_key(key: Any) -> bytes:
    """Encode a supported index key into a deterministic binary representation."""
    if isinstance(key, bool):
        raise TypeError("bool keys are not supported")

    if isinstance(key, int):
        try:
            payload = struct.pack(_INT_FORMAT, key)
        except struct.error as exc:
            raise ValueError("integer key is outside signed 64-bit range") from exc
        return _INT_TAG + payload

    if isinstance(key, str):
        payload = key.encode("utf-8")
        return _STR_TAG + struct.pack(_LENGTH_FORMAT, len(payload)) + payload

    if isinstance(key, tuple):
        encoded_items = [encode_key(item) for item in key]
        parts = [_TUPLE_TAG, struct.pack(_LENGTH_FORMAT, len(encoded_items))]

        for item in encoded_items:
            parts.append(struct.pack(_LENGTH_FORMAT, len(item)))
            parts.append(item)

        return b"".join(parts)

    raise TypeError(f"unsupported index key type: {type(key).__name__}")


def decode_key(data: bytes) -> Any:
    """Decode a key previously produced by :func:`encode_key`."""
    key, offset = _decode_from(data, 0)

    if offset != len(data):
        raise ValueError("unexpected trailing bytes in encoded key")

    return key


def _decode_from(data: bytes, offset: int) -> tuple[Any, int]:
    if offset >= len(data):
        raise ValueError("missing key type tag")

    tag = data[offset : offset + 1]
    offset += 1

    if tag == _INT_TAG:
        end = offset + _INT_SIZE
        if end > len(data):
            raise ValueError("truncated integer key")
        return struct.unpack_from(_INT_FORMAT, data, offset)[0], end

    if tag == _STR_TAG:
        length, offset = _read_length(data, offset)
        end = offset + length
        if end > len(data):
            raise ValueError("truncated string key")

        try:
            value = data[offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid utf-8 string key") from exc

        return value, end

    if tag == _TUPLE_TAG:
        count, offset = _read_length(data, offset)
        items = []

        for _ in range(count):
            item_length, offset = _read_length(data, offset)
            end = offset + item_length

            if end > len(data):
                raise ValueError("truncated composite key")

            item, consumed = _decode_from(data[offset:end], 0)
            if consumed != item_length:
                raise ValueError("invalid composite key item")

            items.append(item)
            offset = end

        return tuple(items), offset

    raise ValueError(f"unknown key type tag: {tag!r}")


def _read_length(data: bytes, offset: int) -> tuple[int, int]:
    end = offset + _LENGTH_SIZE

    if end > len(data):
        raise ValueError("truncated key length")

    return struct.unpack_from(_LENGTH_FORMAT, data, offset)[0], end
