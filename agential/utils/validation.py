"""Validation-related utilities."""
from typing import Dict


def validate_overlapping_keys(dict_1: Dict[str, str], dict_2: Dict[str, str]) -> None:
    """Validates that there are no overlapping keys between two dictionaries.

    Args:
        dict_1 (Dict[str, str]): The first dictionary to check for overlapping keys.
        dict_2 (Dict[str, str]): The second dictionary to check for overlapping keys.

    Raises:
        ValueError: If there are overlapping keys between the two dictionaries.

    Returns:
        None
    """
    overlapping_keys = dict_1.keys() & dict_2.keys()
    if overlapping_keys:
        raise ValueError(f"Overlapping keys detected: {overlapping_keys}")
