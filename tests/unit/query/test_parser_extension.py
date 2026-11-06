import pytest

from engine.common.errors import QueryParseError
from engine.query.ast import CreateExtensionStatement, DropExtensionStatement
from engine.query.parser import parse


def test_create_extension_without_options():
    assert parse("CREATE EXTENSION inverted") == CreateExtensionStatement("INVERTED", ())


def test_create_extension_with_options_and_plugin_alias():
    statement = parse("CREATE PLUGIN inverted WITH (storage_root = '/tmp/p', page_size = 8192)")
    assert statement == CreateExtensionStatement(
        "INVERTED", (("storage_root", "/tmp/p"), ("page_size", 8192))
    )


def test_drop_extension():
    assert parse("DROP EXTENSION inverted") == DropExtensionStatement("INVERTED")


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE EXTENSION",
        "CREATE EXTENSION inverted WITH",
        "CREATE EXTENSION inverted WITH (storage_root)",
        "CREATE EXTENSION inverted WITH (storage_root = '/a', storage_root = '/b')",
        "DROP EXTENSION",
    ],
)
def test_invalid_extension_syntax_is_rejected(sql):
    with pytest.raises(QueryParseError):
        parse(sql)
