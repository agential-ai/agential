"""Utility functions for parsing outputs."""
import re
import string

from typing import List, Optional


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


def clean_str(s: str) -> str:
    """Converts a string with mixed encoding to proper UTF-8 format.

    This function takes a string `s` that may contain Unicode escape sequences and/or Latin-1 encoded characters.
    It processes the string to interpret Unicode escape sequences and correct any Latin-1 encoded parts, returning the string in UTF-8 format.

    Args:
        s (str): The input string potentially containing Unicode escape sequences and Latin-1 encoded characters.

    Returns:
        str: The UTF-8 encoded string with properly interpreted characters.

    Note:
        This function is used in the ReACt implementation.
        This function assumes that the input string is a mix of UTF-8 encoded characters and Unicode escape sequences.
        It may not work as intended if the input string has a different encoding or if it contains characters outside the Latin-1 range.

    See: https://github.com/ysymyth/ReAct/blob/master/wikienv.py.
    """
    return s.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def get_page_obs(page: str, k: Optional[int] = 5) -> str:
    """Extracts and returns the first five sentences from a given text page.

    This function splits a text `page` into paragraphs, then further into sentences.
    It returns the first five sentences of the text, concatenating them into a single string.
    Each sentence is cleaned of leading and trailing spaces.

    Args:
        page (str): The input text page as a string.
        k (int): The number of sentences to return from the page.

    Returns:
        str: A string containing the first five sentences from the input text.

    Note:
        This function is used in the ReACt implementation.
        The function assumes sentences end with a period followed by a space.
        It may not correctly identify sentences in texts with different punctuation styles.

    See: https://github.com/ysymyth/ReAct/blob/master/wikienv.py.
    """
    # Split the page into paragraphs and remove leading/trailing spaces.
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Split paragraphs into sentences and clean each sentence.
    sentences = []
    for p in paragraphs:
        sentences += p.split(". ")
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    # Return the first five sentences joined as a single string.
    return " ".join(sentences[:k])


def construct_lookup_list(keyword: str, page: Optional[str] = None) -> list[str]:
    """Creates a list of sentences from a text page that contain a specified keyword.

    This function takes a keyword and an optional text `page`.
    It finds all sentences in the text that contain the keyword, irrespective of the case.
    If no page is provided, it returns an empty list. The function is case-insensitive.

    Args:
        keyword (str): The keyword to search for in the text.
        page (str, optional): The text page as a string. Defaults to None.

    Returns:
        list[str]: A list of sentences containing the keyword.

    Note:
        This function is used in the ReACt implementation.
        The function assumes sentences are separated by a period followed by a space.
        It may not work correctly for texts with different sentence delimiters.

    See: https://github.com/ysymyth/ReAct/blob/master/wikienv.py.
    """
    # Return an empty list if no page is provided.
    if page is None:
        return []

    # Split the page into paragraphs and remove leading/trailing spaces.
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Split paragraphs into sentences and clean each sentence.
    sentences = []
    for p in paragraphs:
        sentences += p.split(". ")
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    # Filter sentences that contain the keyword, case-insensitive.
    return [p for p in sentences if keyword.lower() in p.lower()]


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
