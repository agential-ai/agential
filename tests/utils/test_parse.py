"""Unit tests for parsing-related functions."""

from agential.utils.parse import (
    remove_newline,
)


def test_remove_newline() -> None:
    """Test remove_newline function."""
    step = "\n  Step with extra spaces and newlines \n"
    assert remove_newline(step) == "Step with extra spaces and newlines"

    # Test with internal newlines.
    step = "Step\nwith\ninternal\nnewlines"
    assert remove_newline(step) == "Stepwithinternalnewlines"

    # Test with a string that doesn't require formatting.
    step = "Already formatted step"
    assert remove_newline(step) == "Already formatted step"
