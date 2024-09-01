"""Unit tests for standard prompting code strategies."""

from agential.core.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_DIRECT
from agential.core.fewshots.mbpp import MBPP_FEWSHOT_EXAMPLES_DIRECT
from agential.llm.llm import BaseLLM, MockLLM, Response
from agential.prompting.standard.output import StandardOutput, StandardStepOutput
from agential.prompting.standard.prompts import (
    STANDARD_INSTRUCTION_HUMANEVAL,
    STANDARD_INSTRUCTION_MBPP,
)
from agential.prompting.standard.strategies.code import StandardCodeStrategy


def test_init() -> None:
    """Tests the initialization of the standard prompting Code strategy."""
    strategy = StandardCodeStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate() -> None:
    """Tests the generate method."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]

    gt_out = StandardOutput(
        answer=[
            [
                "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
            ],
            [
                "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
            ],
        ],
        total_prompt_tokens=40,
        total_completion_tokens=80,
        total_tokens=120,
        total_prompt_cost=6e-05,
        total_completion_cost=0.00015999999999999999,
        total_cost=0.00021999999999999998,
        total_prompt_time=2.0,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                    answer_response=Response(
                        input_text="",
                        output_text="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                StandardStepOutput(
                    answer="def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                    answer_response=Response(
                        input_text="",
                        output_text="def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ],
            [
                StandardStepOutput(
                    answer="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                    answer_response=Response(
                        input_text="",
                        output_text="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                StandardStepOutput(
                    answer="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                    answer_response=Response(
                        input_text="",
                        output_text="from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ],
        ],
    )
    responses = [
        "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
        "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
        "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n# Test cases\nprint(has_close_elements([1.0, 2.0, 3.0], 0.5))  # False\nprint(has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3))  # True",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = StandardCodeStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_HUMANEVAL,
        additional_keys={},
        num_retries=2,
        warming=[0.0, 0.4],
    )
    assert out == gt_out

    gt_out = StandardOutput(
        answer=[
            [
                'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                'def first_repeated_char(s):\n    char_count = {}\n    for char in s:\n        if char in char_count:\n            return char\n        else:\n            char_count[char] = 1\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
            ],
            [
                'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                "def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n",
            ],
        ],
        total_prompt_tokens=40,
        total_completion_tokens=80,
        total_tokens=120,
        total_prompt_cost=6e-05,
        total_completion_cost=0.00015999999999999999,
        total_cost=0.00021999999999999998,
        total_prompt_time=2.0,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                    answer_response=Response(
                        input_text="",
                        output_text='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                StandardStepOutput(
                    answer='def first_repeated_char(s):\n    char_count = {}\n    for char in s:\n        if char in char_count:\n            return char\n        else:\n            char_count[char] = 1\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                    answer_response=Response(
                        input_text="",
                        output_text='def first_repeated_char(s):\n    char_count = {}\n    for char in s:\n        if char in char_count:\n            return char\n        else:\n            char_count[char] = 1\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ],
            [
                StandardStepOutput(
                    answer='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                    answer_response=Response(
                        input_text="",
                        output_text='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
                StandardStepOutput(
                    answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n",
                    answer_response=Response(
                        input_text="",
                        output_text="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ),
            ],
        ],
    )
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
assert first_repeated_char("abc") == None
assert first_repeated_char("123123") == "1\""""

    responses = [
        'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
        'def first_repeated_char(s):\n    char_count = {}\n    for char in s:\n        if char in char_count:\n            return char\n        else:\n            char_count[char] = 1\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
        'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the provided test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
        "def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = StandardCodeStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
        num_retries=2,
        warming=[0.0, 0.4],
    )
    assert out == gt_out
