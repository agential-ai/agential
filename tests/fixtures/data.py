"""Fixtures for loading test data."""

import pytest
import os

@pytest.fixture
def data_dir(pytestconfig) -> str:
    """Dir path to asset."""
    return os.path.join(pytestconfig.rootdir, "tests/assets")

@pytest.fixture
def expel_15_compare_fake_path(data_dir) -> str:
    """Dir path to expel_15_compare_fake experiences."""
    return os.path.join(data_dir, "expel_15_compare_fake.joblib")
