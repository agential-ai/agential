"""Fixtures for loading data-related assets for OSWorldBaseline Agent."""

import os

import pytest


@pytest.fixture
def data_dir(pytestconfig) -> str:
    """Dir path to asset."""
    return os.path.join(pytestconfig.rootdir, "tests/assets")


@pytest.fixture
def osworld_screenshot_path(data_dir: str) -> str:
    """Dir path to OSWorld screenshot."""
    return os.path.join(data_dir, "osworld_baseline", "output_image.jpeg")


@pytest.fixture
def osworld_access_tree(data_dir: str) -> str:
    """Dir path to OSWorld screenshot."""
    return os.path.join(data_dir, "osworld_baseline", "accessibility_tree.txt")
