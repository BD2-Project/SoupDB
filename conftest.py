"""Configuración de pytest."""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SKIP_INTEGRATION") == "1":
        skip = pytest.mark.skip(reason="SKIP_INTEGRATION=1")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
