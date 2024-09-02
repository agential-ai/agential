"""Classification metrics for evaluation."""

from collections import Counter
import re
import string


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


def EM(answer: str, key: str, normalize: bool = True) -> bool:
    """Compares two strings, `answer` and `key`, after normalizing them.

    The Exact Match grading 'metric' compares for an exact match between 2 strings
    after normalization.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
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


def precision(answer: str, key: str, normalize: bool = True) -> float:
    if normalize:
        prediction_tokens = normalize_answer(answer).split()
        ground_truth_tokens = normalize_answer(key).split()
    else:
        prediction_tokens = answer.split()
        ground_truth_tokens = key.split()
        
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    return precision

def recall(answer: str, key: str, normalize: bool = True) -> float:
    if normalize:
        prediction_tokens = normalize_answer(answer).split()
        ground_truth_tokens = normalize_answer(key).split()
    else:
        prediction_tokens = answer.split()
        ground_truth_tokens = key.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall

def f1(answer: str, key: str, normalize: bool = True) -> float:
    if normalize:
        prediction_tokens = normalize_answer(answer).split()
        ground_truth_tokens = normalize_answer(key).split()
    else:
        prediction_tokens = answer.split()
        ground_truth_tokens = key.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1