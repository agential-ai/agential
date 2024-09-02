"""Utility functions for parsing outputs."""

import re
import string


def remove_newline(step: str) -> str:
    """Formats a step string by stripping leading/trailing newlines and spaces, and replacing internal newlines with empty space.

    Args:
        step (str): The step string to be formatted.

    Returns:
        str: The formatted step string.
    """
    return step.strip("\n").strip().replace("\n", "")
