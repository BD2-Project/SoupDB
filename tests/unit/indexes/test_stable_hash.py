import pytest

from engine.indexes._stable_hash import directory_index, stable_hash


def test_hash_is_deterministic() -> None:
    key = ("department", 42)

    assert stable_hash(key) == stable_hash(key)


def test_hash_returns_unsigned_64_bit_integer() -> None:
    value = stable_hash("hello")

    assert 0 <= value < 2**64


def test_different_key_types_do_not_share_encoding_before_hashing() -> None:
    assert stable_hash(1) != stable_hash("1")


def test_composite_keys_are_supported() -> None:
    assert isinstance(stable_hash(("employee", 10)), int)


def test_directory_depth_zero_always_returns_zero() -> None:
    assert directory_index("a", 0) == 0
    assert directory_index("b", 0) == 0


@pytest.mark.parametrize("depth", [1, 2, 4, 8, 16])
def test_directory_index_stays_within_depth(depth: int) -> None:
    result = directory_index("example", depth)

    assert 0 <= result < 2**depth


def test_directory_index_uses_low_hash_bits() -> None:
    key = "example"
    expected = stable_hash(key) & ((1 << 6) - 1)

    assert directory_index(key, 6) == expected


@pytest.mark.parametrize("depth", [-1, 65])
def test_invalid_global_depth_is_rejected(depth: int) -> None:
    with pytest.raises(ValueError):
        directory_index("example", depth)


def test_unsupported_key_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        stable_hash(1.5)
