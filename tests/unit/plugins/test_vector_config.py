import pytest
from soupdb_vector.plugin import IVFPlugin

from engine.common.errors import QueryExecutionError


def test_profiles_json_string_is_normalized():
    plugin = IVFPlugin()
    config = plugin.validate_config(
        {
            "profiles": (
                '{"img-v1": {"kind": "image", "root": "/data", "codebook": "/codebooks/img.npz"}}'
            )
        }
    )
    assert config["profiles"] == {
        "img-v1": {
            "kind": "image",
            "root": "/data",
            "codebook": "/codebooks/img.npz",
        }
    }


def test_invalid_profiles_json_is_rejected():
    with pytest.raises(QueryExecutionError, match="valid JSON"):
        IVFPlugin().validate_config({"profiles": "{not-json"})


def test_profile_without_root_or_codebook_is_rejected():
    with pytest.raises(QueryExecutionError, match="root"):
        IVFPlugin().validate_config({"profiles": {"img": {"kind": "image", "codebook": "/c.npz"}}})
    with pytest.raises(QueryExecutionError, match="codebook"):
        IVFPlugin().validate_config({"profiles": {"img": {"kind": "image", "root": "/data"}}})
