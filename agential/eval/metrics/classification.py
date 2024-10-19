"""Classification metrics for evaluation."""

import math
import re
import string

from collections import Counter
from typing import List, Union

from thefuzz import fuzz

from agential.core.llm import BaseLLM

LLM_AS_JUDGE_EVAL_FEWSHOT_EXAMPLES = """Question: What was the name of the first successful moon landing mission?
Reference Answers: Apollo 11
Predicted Answers: Apollo mission
Score: 0

---

Question: What is the tallest mountain in the world?
Reference Answers: Mt. Everest
Predicted Answers: The tallest mountain is Mount Everest
Score: 1

---

Question: What is the largest city in the United States by population?
Reference Answers: New York City | NYC | NYC, New York
Predicted Answers: Washington, D.C.
Score: 0

---

Question: What is the capital city of France?
Reference Answers: The capital of France is Paris | Paris
Predicted Answers: Paris
Score: 1

---

Question: Which element is essential for human respiration?
Reference Answers: Oxygen
Predicted Answers: Oxygen
Score: 1"""


LLM_AS_JUDGE_EVAL_INSTRUCTION = """You are an expert annotator and evaluator. Your job is to compare the reference answer(s) and the predicted answer and determine whether the predicted answer is correct.
Output 1 if the predicted answer is semantically similar to any of the reference answer(s) delimited by a | and correctly answers the question otherwise output 0.

{examples}
(END OF EXAMPLES)

Question: {question}
Reference Answers: {key}
Predicted Answers: {answer}
Score: """


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


def llm_as_judge_eval(
    llm: BaseLLM,
    question: str,
    answer: Union[str, List[str]],
    key: Union[str, List[str]],
    examples: str = LLM_AS_JUDGE_EVAL_FEWSHOT_EXAMPLES,
    prompt: str = LLM_AS_JUDGE_EVAL_INSTRUCTION,
    with_em: bool = True,
) -> bool:
    """Determines whether to use LLM as a judge for evaluation.

    Args:
        llm (BaseLLM): The language model to be used for evaluation.
        question (str): The question to be evaluated.
        answer (Union[str, List[str]]): A string or a list of strings to be compared with `key`.
        key (Union[str, List[str]]): A string or a list of strings to be compared with `answer`.
        examples (str): A string of examples to be used in the prompt. Defaults to LLM_AS_JUDGE_EVAL_FEWSHOT_EXAMPLES.
        prompt (str): The prompt to be used for evaluation. Defaults to LLM_AS_JUDGE_EVAL_INSTRUCTION.
        with_em (bool): Whether to check with exact match before prompting to save costs. Defaults to True.

    Returns:
        bool: True if the answer matches the key, otherwise False.
    """
    if answer is None or answer == "":
        return False

    if isinstance(answer, list):
        answer = "| ".join(answer)

    if isinstance(key, list):
        key = "| ".join(key)

    if with_em and normalize_answer(answer) == normalize_answer(key):
        return True

    prompt = prompt.format(examples=examples, question=question, key=key, answer=answer)
    out = llm(prompt)

    integer_pattern = r"\b\d+\b"
    match = re.search(integer_pattern, out.output_text)
    if match:
        return int(match.group(0)) == 1
    return False


def EM(
    answer: str,
    key: str,
    normalize: bool = True,
    is_numeric: bool = False,
) -> bool:
    """Compares two strings, `answer` and `key`, after normalizing them.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Only applies to is_numeric=False. Defaults to True.
        is_numeric (bool): A boolean indicating if the answer and key are numeric values. Defaults to False.

    Returns:
        bool: True if the normalized `answer` and `key` match, else False.
    """
    if answer is None:
        return False

    if not is_numeric:
        answer = normalize_answer(answer) if normalize else answer
        key = normalize_answer(key) if normalize else key

        return answer == key
    else:
        try:
            return math.isclose(float(parse_first_number(answer)), float(key))
        except:
            return answer == key


def fuzzy_EM(
    answer: str, key: str, normalize: bool = True, fuzzy_threshold: float = 0.95
) -> bool:
    """Compares two strings, `answer` and `key`, after normalizing them.

    Args:
        answer (str): A string to be compared with `key`. Can be "".
        key (str): A string to be compared with `answer`.
        normalize (bool): If True, then normalize answer and key. Defaults to True.
        fuzzy_threshold (float): A float indicating the threshold for fuzzy matching. Defaults to 0.95.

    Returns:
        bool: True if the normalized `answer` and `key` match, else False.
    """
    answer = normalize_answer(answer) if normalize else answer
    key = normalize_answer(key) if normalize else key

    ratio1 = fuzz.partial_ratio(answer, key)
    ratio2 = fuzz.token_set_ratio(answer, key)
    ratio3 = fuzz.partial_token_sort_ratio(answer, key)
    above_threshold = all(
        [ratio / 100 > fuzzy_threshold for ratio in (ratio1, ratio2, ratio3)]
    )

    return answer == key or above_threshold


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
