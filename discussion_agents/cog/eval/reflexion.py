"""Evaluation module for Reflexion."""

from discussion_agents.utils.parse import normalize_answer


def EM(answer: str, key: str) -> bool:
    """Compares two strings, `answer` and `key`, after normalizing them.

    The Exact Match grading 'metric' compares for an exact match between 2 strings
    after normalization.

    Args:
        answer (str): A string to be compared with `key`.
        key (str): A string to be compared with `answer`.

    Returns:
        bool: True if the normalized `answer` and `key` match, else False.
    """
    return normalize_answer(answer) == normalize_answer(key)
