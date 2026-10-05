import pytest

from engine.indexes._key_codec import decode_key, encode_key


@pytest.mark.parametrize(
    "key",
    [
        0,
        1,
        -1,
        123456,
        "hello",
        "",
        "árbol",
        (1, "a"),
        ("department", 42),
        (1, ("nested", 2)),
    ],
)
def test_key_roundtrip(key) -> None:
    assert decode_key(encode_key(key)) == key


def test_encoding_is_deterministic() -> None:
    key = ("employee", 42)

    assert encode_key(key) == encode_key(key)


def test_different_types_have_different_encodings() -> None:
    assert encode_key(1) != encode_key("1")


def test_bool_is_rejected() -> None:
    with pytest.raises(TypeError):
        encode_key(True)


def test_unsupported_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        encode_key(1.5)


def test_integer_outside_signed_64_bit_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        encode_key(2**63)


def test_decode_rejects_unknown_tag() -> None:
    with pytest.raises(ValueError):
        decode_key(b"x")


def test_decode_rejects_trailing_bytes() -> None:
    encoded = encode_key(10)

    with pytest.raises(ValueError):
        decode_key(encoded + b"x")


def test_decode_rejects_truncated_integer() -> None:
    with pytest.raises(ValueError):
        decode_key(b"i")


def test_decode_rejects_truncated_string() -> None:
    encoded = encode_key("hello")

    with pytest.raises(ValueError):
        decode_key(encoded[:-1])
