"""Configuración de pytest.

Los tests marcados con ``integration`` (intensivos) corren por defecto — el CI
los incluye. Para desactivarlos:

- ``SKIP_INTEGRATION=1 uv run pytest``
- o ``uv run pytest -m "not integration"``
- o solo integración: ``uv run pytest -m integration``
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SKIP_INTEGRATION") == "1":
        skip = pytest.mark.skip(reason="SKIP_INTEGRATION=1")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
