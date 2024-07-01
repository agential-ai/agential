"""Evaluation module for Reflexion."""

from agential.utils.parse import normalize_answer


def EM(answer: str, key: str, normalize: bool = True) -> bool:
    """Compares two strings, `answer` and `key`, after normalizing them.

    The Exact Match grading 'metric' compares for an exact match between 2 strings
    after normalization.

    Args:
        answer (str): A string to be compared with `key`.
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Defaults to True.

    Returns:
        bool: True if the normalized `answer` and `key` match, else False.
    """
    if answer is None:
        return False

    if not normalize:
        return answer == key
    return normalize_answer(answer) == normalize_answer(key)
