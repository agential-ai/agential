"""Utility functions for parsing outputs."""

import re
import string

from typing import List, Optional, Tuple


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

    Example:
        input_text = "1. Item 1\n2. Item 2\n3. Item 3\n\n4. Item 4"
        parsed_list = GenerativeAgentMemory._parse_list(input_text)
        # 'parsed_list' contains ["Item 1", "Item 2", "Item 3", "Item 4"]

    Note:
        - This method is useful for parsing structured text into a list of items.
        - It removes leading numbers and periods often used in numbered lists.
    """
    lines = re.split(r"\n", text.strip())
    lines = [line for line in lines if line.strip()]  # Remove empty lines.
    lines = [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]
    return lines


def parse_numbered_list(text: str) -> List[str]:
    r"""Parse a numbered list from a given text and return a list of list items.

    This function extracts the content following the last ")" character, removes any trailing
    commas or periods, and trims leading/trailing spaces from each line in the input text.

    Args:
        text (str): The input text containing a numbered list.

    Returns:
        List[str]: A list of strings from the numbered list.

    Example:
        input_text = "1) Item One.\n2) Item Two.\n3) Item Three,\n"
        parsed_list = parse_numbered_list(input_text)
        # Result: ["Item One", "Item Two", "Item Three"]
    """
    lines = parse_list(text)
    lines = [s.split(")")[-1].rstrip(",.").strip() for s in lines]
    return lines


def remove_name(text: str, name: str) -> str:
    """Remove a specified name prefix from the beginning of each line in the text.

    This function removes the specified 'name' prefix followed by a space from the
    beginning of each line in the input text.

    Args:
        text (str): The input text containing lines with name prefixes.
        name (str): The name prefix to remove from each line.

    Returns:
        str: The text with the specified name prefix removed from each line.

    Example:
        input_text = "John Smith"
        clean_text = remove_name(input_text, "John")
        # Result: "Smith"
    """
    lines = re.sub(f"^{name} ", "", text.strip()).strip()
    return lines


def remove_newline(step: str) -> str:
    """Formats a step string by stripping leading/trailing newlines and spaces, and replacing internal newlines with empty space.

    Args:
        step (str): The step string to be formatted.

    Returns:
        str: The formatted step string.
    """
    return step.strip("\n").strip().replace("\n", "")


def remove_articles(text: str) -> str:
    """Remove articles ('a', 'an', 'the') from the text.

    Args:
        text (str): The input string from which articles need to be removed.

    Returns:
        str: The modified string with articles removed.
    """
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    """Fix any irregular white spaces in the text.

    Args:
        text (str): The input string with potential irregular white spaces.

    Returns:
        str: The modified string with normalized white spaces.
    """
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    """Remove punctuation from the text.

    Args:
        text (str): The input string from which punctuation needs to be removed.

    Returns:
        str: The modified string with punctuation removed.
    """
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    """Normalize an answer by removing articles, fixing white spaces, and removing punctuation.

    Args:
        s (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    return white_space_fix(remove_articles(remove_punc(s.lower())))
