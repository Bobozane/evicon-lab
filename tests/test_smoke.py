"""Minimal package-level regression check for the project skeleton."""

import evicon


def test_package_is_importable() -> None:
    assert evicon.__version__ == "0.1.0"

