"""Utility functions for parsing outputs."""
import re

from typing import List


def parse_list(text: str) -> List[str]:
    r"""Parse a newline-separated string into a list of strings.

    This static method takes a string that contains multiple lines separated by
    newline characters and parses it into a list of strings. It removes any empty
    lines and also removes any leading numbers followed by a period (commonly used
    in numbered lists).

    Args:
        text (str): The input string containing newline-separated lines.

    Returns:
        List[str]: A list of strings parsed from the input text.

    Example usage:
        input_text = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
        parsed_list = GenerativeAgentMemory._parse_list(input_text)
        # 'parsed_list' contains ["Item 1", "Item 2", "Item 3", "Item 4"]

    Note:
        - This method is useful for parsing structured text into a list of items.
        - It removes leading numbers and periods often used in numbered lists.
    """
    lines = re.split(r"\n", text.strip())
    lines = [line for line in lines if line.strip()]  # remove empty lines
    result = [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]
    return result

def parse_numbered_list(text: str) -> List[str]:
    result = parse_list(text)
    result = [s.split(")")[-1].rstrip(",.").strip() for s in result]
    return result

def remove_name(text: str, name: str) -> str:
    result = re.sub(f"^{name} ", "", text.strip()).strip()
    return result