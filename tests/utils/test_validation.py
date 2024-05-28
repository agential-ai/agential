"""Unit tests for validation-related functions."""

import pytest

from agential.utils.validation import validate_overlapping_keys


def test_validate_overlapping_keys() -> None:
    """Test validate_overlapping_keys function."""
    # Case 1: No overlap.
    dict_1 = {"a": "1", "b": "2"}
    dict_2 = {"c": "3", "d": "4"}
    # This should not raise any exception.
    validate_overlapping_keys(dict_1, dict_2)

    # Case 2: With overlap.
    dict_1 = {"a": "1", "b": "2"}
    dict_2 = {"b": "3", "d": "4"}
    # This should raise a ValueError due to the overlapping key "b".
    with pytest.raises(ValueError) as excinfo:
        validate_overlapping_keys(dict_1, dict_2)
    assert "Overlapping keys detected: {'b'}" in str(excinfo.value)

    # Case 3: Empty dictionaries.
    dict_1 = {}
    dict_2 = {}
    # This should not raise any exception as both dictionaries are empty.
    validate_overlapping_keys(dict_1, dict_2)

    # Case 4: One empty dictionary.
    dict_1 = {"a": "1", "b": "2"}
    dict_2 = {}
    # This should not raise any exception as one of the dictionaries is empty.
    validate_overlapping_keys(dict_1, dict_2)

    # Case 5: Multiple overlaps.
    dict_1 = {"a": "1", "b": "2", "c": "3"}
    dict_2 = {"b": "3", "c": "4", "d": "5"}
    # This should raise a ValueError due to the overlapping keys "b" and "c".
    with pytest.raises(ValueError):
        validate_overlapping_keys(dict_1, dict_2)
