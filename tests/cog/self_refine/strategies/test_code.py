"""Unit tests for Self-Refine code strategies."""

from agential.core.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_POT
from agential.agent.self_refine.output import SelfRefineOutput, SelfRefineStepOutput
from agential.agent.self_refine.prompts import (
    HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
    HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
)
from agential.agent.self_refine.strategies.code import (
    SelfRefineCodeStrategy,
    SelfRefineHEvalStrategy,
    SelfRefineMBPPStrategy,
)
from agential.llm.llm import MockLLM, Response


def test_init() -> None:
    """Test SelfRefineCodeStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineCodeStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy.testing == False
    assert strategy._prev_answer == ""
    assert strategy.patience_counter == 0


def test_generate() -> None:
    """Test SelfRefineCodeStrategy generate."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"

    gt_out = SelfRefineOutput(
        answer='from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)',
        total_prompt_tokens=60,
        total_completion_tokens=120,
        total_tokens=180,
        total_prompt_cost=9e-05,
        total_completion_cost=0.00023999999999999998,
        total_cost=0.00033,
        total_prompt_time=3.0,
        total_time=0.5,
        additional_info=[
            SelfRefineStepOutput(
                answer="def has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])",
                critique="The implementation of the `has_close_elements` function is missing the return statement. The function itself iterates through pairs of numbers in the list and checks if the absolute difference between them falls below the given threshold. However, it lacks a return statement to provide the final result of whether any two numbers are closer than the threshold.\n\nTo fix this issue, the `has_close_elements` function should include a return statement at the end to return the boolean result after checking all pairs of numbers. Here is the revised code with the return statement added:\n\n```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```\n\nBy adding `return` at the beginning of this line, the function will correctly return `True` if any two numbers in the list are closer than the threshold and `False` otherwise.",
                answer_response=Response(
                    input_text="",
                    output_text="```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="The implementation of the `has_close_elements` function is missing the return statement. The function itself iterates through pairs of numbers in the list and checks if the absolute difference between them falls below the given threshold. However, it lacks a return statement to provide the final result of whether any two numbers are closer than the threshold.\n\nTo fix this issue, the `has_close_elements` function should include a return statement at the end to return the boolean result after checking all pairs of numbers. Here is the revised code with the return statement added:\n\n```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```\n\nBy adding `return` at the beginning of this line, the function will correctly return `True` if any two numbers in the list are closer than the threshold and `False` otherwise.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            SelfRefineStepOutput(
                answer='from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])',
                critique="The implementation of the `has_close_elements` function has a logical error in the condition used to check if any two numbers in the list are closer to each other than the given threshold.\n\n1. The primary issue lies in the condition used within the `any` function:\n   ```python\n   return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n   ```\n   This condition is meant to check the absolute difference between every pair of numbers (a, b) in the list and return `True` if the difference is less than the threshold for any pair. However, the current implementation fails to consider the case when `a` and `b` are the same number (which implies a difference of 0), potentially leading to incorrect results.\n\n2. In the given test cases:\n   - The case with the numbers `[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]` and threshold `0.3` is expected to return `True` since `2.2` and `3.9` are closer than the threshold.\n   - The case with the same numbers but a threshold of `0.05` is expected to return `False` since the closest numbers are not within the threshold.\n\n3. To address the issue, the condition should be modified to exclude comparing the same number with itself:\n   ```python\n   return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n   ```\n\nBy adjusting the condition to exclude comparing a number with itself, the function will correctly identify cases where two distinct numbers in the list are closer to each other than the specified threshold, providing accurate outcomes.",
                answer_response=Response(
                    input_text="",
                    output_text='```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```',
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="The implementation of the `has_close_elements` function has a logical error in the condition used to check if any two numbers in the list are closer to each other than the given threshold.\n\n1. The primary issue lies in the condition used within the `any` function:\n   ```python\n   return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n   ```\n   This condition is meant to check the absolute difference between every pair of numbers (a, b) in the list and return `True` if the difference is less than the threshold for any pair. However, the current implementation fails to consider the case when `a` and `b` are the same number (which implies a difference of 0), potentially leading to incorrect results.\n\n2. In the given test cases:\n   - The case with the numbers `[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]` and threshold `0.3` is expected to return `True` since `2.2` and `3.9` are closer than the threshold.\n   - The case with the same numbers but a threshold of `0.05` is expected to return `False` since the closest numbers are not within the threshold.\n\n3. To address the issue, the condition should be modified to exclude comparing the same number with itself:\n   ```python\n   return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n   ```\n\nBy adjusting the condition to exclude comparing a number with itself, the function will correctly identify cases where two distinct numbers in the list are closer to each other than the specified threshold, providing accurate outcomes.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            SelfRefineStepOutput(
                answer='from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)',
                critique="The `has_close_elements` function has a logical issue in the way it checks for close elements in the given list of numbers. \n\nHere's the problem: \n\nThe function uses a nested loop with list comprehension to iterate over every pair of numbers in the list and checks if the absolute difference between them is less than the threshold. However, this implementation incorrectly compares every number with every other number in the list, including comparing a number with itself (i.e., when `i == j`). This leads to false positives when the difference between a number and itself is below the threshold, which should not be considered a pair of close elements.\n\nTo fix this issue, you should modify the condition to exclude comparisons between the same numbers by adding an additional check to ensure `i != j` before comparing the two numbers:\n```python\nreturn any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```\n\nThis adjustment ensures that each number is only compared with distinct numbers in the list, eliminating false positives that would occur when comparing a number with itself. \n\nAfter making this change, the function should correctly check for close elements in the list according to the provided threshold.",
                answer_response=Response(
                    input_text="",
                    output_text='```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```',
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="The `has_close_elements` function has a logical issue in the way it checks for close elements in the given list of numbers. \n\nHere's the problem: \n\nThe function uses a nested loop with list comprehension to iterate over every pair of numbers in the list and checks if the absolute difference between them is less than the threshold. However, this implementation incorrectly compares every number with every other number in the list, including comparing a number with itself (i.e., when `i == j`). This leads to false positives when the difference between a number and itself is below the threshold, which should not be considered a pair of close elements.\n\nTo fix this issue, you should modify the condition to exclude comparisons between the same numbers by adding an additional check to ensure `i != j` before comparing the two numbers:\n```python\nreturn any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```\n\nThis adjustment ensures that each number is only compared with distinct numbers in the list, eliminating false positives that would occur when comparing a number with itself. \n\nAfter making this change, the function should correctly check for close elements in the list according to the provided threshold.",
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
    )
    responses = [
        "```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```",
        "The implementation of the `has_close_elements` function is missing the return statement. The function itself iterates through pairs of numbers in the list and checks if the absolute difference between them falls below the given threshold. However, it lacks a return statement to provide the final result of whether any two numbers are closer than the threshold.\n\nTo fix this issue, the `has_close_elements` function should include a return statement at the end to return the boolean result after checking all pairs of numbers. Here is the revised code with the return statement added:\n\n```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```\n\nBy adding `return` at the beginning of this line, the function will correctly return `True` if any two numbers in the list are closer than the threshold and `False` otherwise.",
        '```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n```',
        "The implementation of the `has_close_elements` function has a logical error in the condition used to check if any two numbers in the list are closer to each other than the given threshold.\n\n1. The primary issue lies in the condition used within the `any` function:\n   ```python\n   return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i+1:])\n   ```\n   This condition is meant to check the absolute difference between every pair of numbers (a, b) in the list and return `True` if the difference is less than the threshold for any pair. However, the current implementation fails to consider the case when `a` and `b` are the same number (which implies a difference of 0), potentially leading to incorrect results.\n\n2. In the given test cases:\n   - The case with the numbers `[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]` and threshold `0.3` is expected to return `True` since `2.2` and `3.9` are closer than the threshold.\n   - The case with the same numbers but a threshold of `0.05` is expected to return `False` since the closest numbers are not within the threshold.\n\n3. To address the issue, the condition should be modified to exclude comparing the same number with itself:\n   ```python\n   return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n   ```\n\nBy adjusting the condition to exclude comparing a number with itself, the function will correctly identify cases where two distinct numbers in the list are closer to each other than the specified threshold, providing accurate outcomes.",
        '```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```',
        "The `has_close_elements` function has a logical issue in the way it checks for close elements in the given list of numbers. \n\nHere's the problem: \n\nThe function uses a nested loop with list comprehension to iterate over every pair of numbers in the list and checks if the absolute difference between them is less than the threshold. However, this implementation incorrectly compares every number with every other number in the list, including comparing a number with itself (i.e., when `i == j`). This leads to false positives when the difference between a number and itself is below the threshold, which should not be considered a pair of close elements.\n\nTo fix this issue, you should modify the condition to exclude comparisons between the same numbers by adding an additional check to ensure `i != j` before comparing the two numbers:\n```python\nreturn any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```\n\nThis adjustment ensures that each number is only compared with distinct numbers in the list, eliminating false positives that would occur when comparing a number with itself. \n\nAfter making this change, the function should correctly check for close elements in the list according to the provided threshold.",
        '```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for j, b in enumerate(numbers) if i != j)\n```',
    ]

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineCodeStrategy(llm=llm, testing=True)

    out = strategy.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=SELF_REFINE_INSTRUCTION_HUMANEVAL,
        critique_examples=HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
        critique_prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
        refine_examples=HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt=SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
        additional_keys={},
        critique_additional_keys={"tests": tests},
        refine_additional_keys={"tests": tests},
        max_interactions=3,
        reset=True,
    )
    assert out == gt_out


def test_generate_answer() -> None:
    """Tests SelfRefineCodeStrategy generate."""
    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=[
            'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False\n'
        ],
    )

    strategy = SelfRefineCodeStrategy(llm=llm)

    question = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'

    answer, out = strategy.generate_answer(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=SELF_REFINE_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert (
        answer
        == 'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False'
    )
    assert out == Response(
        input_text="",
        output_text='from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False\n',
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_critique() -> None:
    """Tests SelfRefineCodeStrategy generate_critique."""
    gt_critique = "The function incorrectly returns True for a list without duplicates due to a logical error in the comparison operation. For example, with `names_list = ['Alice', 'Bob', 'Charlie', 'Dave']`, the function still returns True. The line `return len(names_list) != len(set(names_list)) - 1` checks for duplicates by comparing the list's length with the set's length (which removes duplicates), subtracting one to allow exactly one duplicate. However, this logic is flawed. The subtraction causes a false positive for duplicates when there is any unique item, as it misinterprets the size difference. The function thus fails due to a critical error in this comparison, leading to incorrect duplicate identification."
    responses = [
        "The function incorrectly returns True for a list without duplicates due to a logical error in the comparison operation. For example, with `names_list = ['Alice', 'Bob', 'Charlie', 'Dave']`, the function still returns True. The line `return len(names_list) != len(set(names_list)) - 1` checks for duplicates by comparing the list's length with the set's length (which removes duplicates), subtracting one to allow exactly one duplicate. However, this logic is flawed. The subtraction causes a false positive for duplicates when there is any unique item, as it misinterprets the size difference. The function thus fails due to a critical error in this comparison, leading to incorrect duplicate identification."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineCodeStrategy(llm=llm)
    question = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'
    answer = 'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False'
    tests = "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n\ncheck(has_close_elements)"

    critique, finished, out = strategy.generate_critique(
        question=question,
        examples=HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": tests},
    )

    assert critique == gt_critique
    assert strategy._prev_answer == answer
    assert strategy.patience_counter == 0
    assert finished == False
    assert out == Response(
        input_text="",
        output_text="The function incorrectly returns True for a list without duplicates due to a logical error in the comparison operation. For example, with `names_list = ['Alice', 'Bob', 'Charlie', 'Dave']`, the function still returns True. The line `return len(names_list) != len(set(names_list)) - 1` checks for duplicates by comparing the list's length with the set's length (which removes duplicates), subtracting one to allow exactly one duplicate. However, this logic is flawed. The subtraction causes a false positive for duplicates when there is any unique item, as it misinterprets the size difference. The function thus fails due to a critical error in this comparison, leading to incorrect duplicate identification.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test early stopping.
    gt_critique = "The function incorrectly returns True for a list without duplicates due to a logical error in the comparison operation. For example, with `names_list = ['Alice', 'Bob', 'Charlie', 'Dave']`, the function still returns True. The line `return len(names_list) != len(set(names_list)) - 1` checks for duplicates by comparing the list's length with the set's length (which removes duplicates), subtracting one to allow exactly one duplicate. However, this logic is flawed. The subtraction causes a false positive for duplicates when there is any unique item, as it misinterprets the size difference. The function thus fails due to a critical error in this comparison, leading to incorrect duplicate identification."
    responses = [
        "The function incorrectly returns True for a list without duplicates due to a logical error in the comparison operation. For example, with `names_list = ['Alice', 'Bob', 'Charlie', 'Dave']`, the function still returns True. The line `return len(names_list) != len(set(names_list)) - 1` checks for duplicates by comparing the list's length with the set's length (which removes duplicates), subtracting one to allow exactly one duplicate. However, this logic is flawed. The subtraction causes a false positive for duplicates when there is any unique item, as it misinterprets the size difference. The function thus fails due to a critical error in this comparison, leading to incorrect duplicate identification."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)

    question = 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n'
    answer = 'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False'
    tests = "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n\ncheck(has_close_elements)"
    strategy = SelfRefineCodeStrategy(llm=llm, patience=1)
    strategy._prev_answer = 'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False'

    critique, finished, out = strategy.generate_critique(
        question=question,
        examples=HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": tests},
    )
    assert critique == gt_critique
    assert strategy.patience_counter == 1
    assert (
        strategy._prev_answer
        == 'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False'
    )
    assert finished
    assert out == Response(
        input_text="",
        output_text="The function incorrectly returns True for a list without duplicates due to a logical error in the comparison operation. For example, with `names_list = ['Alice', 'Bob', 'Charlie', 'Dave']`, the function still returns True. The line `return len(names_list) != len(set(names_list)) - 1` checks for duplicates by comparing the list's length with the set's length (which removes duplicates), subtracting one to allow exactly one duplicate. However, this logic is flawed. The subtraction causes a false positive for duplicates when there is any unique item, as it misinterprets the size difference. The function thus fails due to a critical error in this comparison, leading to incorrect duplicate identification.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineCodeStrategy update_answer_based_on_critique."""
    gt_answer = 'from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    responses = [
        '```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```'
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineCodeStrategy(llm=llm, patience=2)
    new_answer, out = strategy.update_answer_based_on_critique(
        question="", examples="", answer="", critique="", prompt="", additional_keys={}
    )
    assert new_answer == gt_answer
    assert out == Response(
        input_text="",
        output_text='```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```',
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_halting_condition() -> None:
    """Tests SelfRefineCodeStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineCodeStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition(False) == False

    # Simulate the halting condition being met.
    assert strategy.halting_condition(True)


def test_reset() -> None:
    """Tests SelfRefineCodeStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineCodeStrategy(llm=llm, patience=2)

    strategy._prev_answer = "result = 42"
    strategy.patience_counter = 1
    strategy.reset()
    assert strategy._prev_answer == ""
    assert strategy.patience_counter == 0


def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    assert isinstance(SelfRefineHEvalStrategy(llm=llm), SelfRefineHEvalStrategy)
    assert isinstance(SelfRefineMBPPStrategy(llm=llm), SelfRefineMBPPStrategy)
