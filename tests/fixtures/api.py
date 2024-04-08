"""Fixtures for API keys."""

import os

import pytest


@pytest.fixture
def google_api_key() -> str:
    """Fixture to retrieve the google_api_key from environment variables."""
    key = os.getenv("GOOGLE_API_KEY")
    if key is None:
        pytest.fail(
            "GOOGLE_API_KEY not set in the environment variables", pytrace=False
        )
    return key


@pytest.fixture
def google_cse_id() -> str:
    """Fixture to retrieve the google_cse_id from environment variables."""
    key = os.getenv("GOOGLE_CSE_ID")
    if key is None:
        pytest.fail("GOOGLE_CSE_ID not set in the environment variables", pytrace=False)
    return key
