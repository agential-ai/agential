"""Classification metrics for evaluation."""

import math
import re
import string

from collections import Counter

from thefuzz import fuzz


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


def parse_first_number(s: str) -> str:
    """Parses the first number out of the string.

    Args:
        s (str): The string.

    Returns:
        str: The parsed number as a string or "".
    """
    number_pattern = r"[-+]?\d*\.?\d+"
    match = re.search(number_pattern, s)
    if match:
        return str(float(match.group(0)))
    else:
        return ""


def EM(
    answer: str,
    key: str,
    normalize: bool = True,
    is_numeric: bool = False,
    fuzzy: bool = True,
    fuzzy_threshold: float = 0.90,
) -> bool:
    """Compares two strings, `answer` and `key`, after normalizing them.

    The Exact Match grading 'metric' compares for an exact match between 2 strings
    after normalization. This metric also supports numeric comparisons and fuzzy matching.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Only applies to is_numeric=False. Defaults to True.
        is_numeric (bool): A boolean indicating if the answer and key are numeric values. Defaults to False.
        fuzzy (bool): A boolean indicating if the answer and key are fuzzy matches. Only applies to is_numeric=False. Defaults to True.
        fuzzy_threshold (float): A float indicating the threshold for fuzzy matching. Only applies to is_numeric=False. Defaults to 0.90.

    Returns:
        bool: True if the normalized `answer` and `key` match, else False.
    """
    if answer is None:
        return False

    if not is_numeric:
        answer = normalize_answer(answer) if normalize else answer
        key = normalize_answer(key) if normalize else key

        if fuzzy:
            ratio1 = fuzz.partial_ratio(answer, key)
            ratio2 = fuzz.token_set_ratio(answer, key)
            ratio3 = fuzz.partial_token_sort_ratio(answer, key)
            above_threshold = all(
                [ratio / 100 > fuzzy_threshold for ratio in (ratio1, ratio2, ratio3)]
            )
        else:
            above_threshold = False

        return answer == key or (above_threshold and key in answer)
    else:
        try:
            return math.isclose(float(parse_first_number(answer)), float(key))
        except:
            return answer == key


def precision(answer: str, key: str, normalize: bool = True) -> float:
    """Computes the precision score between two strings, `answer` and `key`.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Defaults to True.

    Returns:
        float: The precision score between `answer` and `key`.
    """
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
    """Computes the recall score between two strings, `answer` and `key`.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Defaults to True.

    Returns:
        float: The recall score between `answer` and `key`.
    """
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
    """Computes the F1 score between two strings, `answer` and `key`.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Defaults to True.

    Returns:
        float: The F1 score between `answer` and `key`.
    """
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
