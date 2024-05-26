"""Unit tests for CRITIC code strategies."""

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.prompts.critic import (
    # HumanEval.
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,

    # MBPP.
    CRITIC_POT_INSTRUCTION_MBPP,
    MBPP_FEWSHOT_EXAMPLES_POT,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
)
from agential.cog.strategies.critic.code_strategy import (
    CriticCodeStrategy,
    CritMBPPCodeStrategy,
    CritHEvalCodeStrategy
)


def test_init() -> None:
    """Test CriticCodeStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = CriticCodeStrategy(llm=llm)
    assert strategy.llm == llm
    assert not strategy._halt


def test_generate() -> None:
    """Tests CriticCodeStrategy generate."""
    inst = {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}
    question = inst['prompt']

    gt_result = '    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    responses = [
        '```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    result = strategy.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert result == gt_result
    assert strategy._halt is False


def test_generate_critique() -> None:
    """Tests CriticCodeStrategy generate_critique."""
    inst = {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}
    question = inst['prompt']
    answer = '```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```'
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"

    # Test no tool.
    gt_critique = 'The provided code snippet defines a function `has_close_elements` that aims to check if there are any two numbers in a given list that are closer to each other than a specified threshold. The function iterates over all pairs of numbers in the list and returns `True` if it finds a pair that meets the condition.\n\nHowever, there is a logical flaw in the comparison of the absolute difference between numbers. The current implementation checks `abs(numbers[i] - numbers[j]) < threshold` for closeness. This condition is incorrect as it only considers the absolute difference between two numbers but does not account for the direction or relative positions of the numbers.\n\nFor example, if we have `numbers = [1.0, 2.0, 3.0]` and `threshold = 0.5`, the function would incorrectly return `True` even though the numbers in the list are not closer to each other than the threshold.\n\nTo correct this issue, the function should compare the absolute difference between numbers to the threshold value while also considering their relative positions. One way to achieve this is by sorting the list of numbers before comparing adjacent pairs, ensuring that the comparison is done in a meaningful way that respects the order of the elements.\n\n'
    responses = [
        'The provided code snippet defines a function `has_close_elements` that aims to check if there are any two numbers in a given list that are closer to each other than a specified threshold. The function iterates over all pairs of numbers in the list and returns `True` if it finds a pair that meets the condition.\n\nHowever, there is a logical flaw in the comparison of the absolute difference between numbers. The current implementation checks `abs(numbers[i] - numbers[j]) < threshold` for closeness. This condition is incorrect as it only considers the absolute difference between two numbers but does not account for the direction or relative positions of the numbers.\n\nFor example, if we have `numbers = [1.0, 2.0, 3.0]` and `threshold = 0.5`, the function would incorrectly return `True` even though the numbers in the list are not closer to each other than the threshold.\n\nTo correct this issue, the function should compare the absolute difference between numbers to the threshold value while also considering their relative positions. One way to achieve this is by sorting the list of numbers before comparing adjacent pairs, ensuring that the comparison is done in a meaningful way that respects the order of the elements.\n\nHere\'s a revised implementation of the `has_close_elements` function:\n\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in the given list of numbers, any two numbers are closer to each other than the given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    numbers.sort()\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i+1]) < threshold:\n            return True\n    return False\n```\n\nThis implementation addresses the issue by sorting the list of numbers before comparing adjacent elements. This ensures that the function correctly identifies pairs of numbers that are closer to each other than the specified threshold.'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticCodeStrategy(llm=llm)


    critique, external_tool_info = strategy.generate_critique(
        idx=0,
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": tests},
        use_tool=False,
        max_interactions=7
    )
    assert critique == gt_critique
    assert external_tool_info == {}
    assert not strategy._halt

    # Test no tests error.
    with pytest.raises(ValueError):
        critique, external_tool_info = strategy.generate_critique(
            idx=0,
            question=question,
            examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
            answer=answer,
            critique="",
            prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
            additional_keys={},
            use_tool=True,
            max_interactions=7
        )

    # Test with tool.
    gt_critique = ""
    responses = [
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticCodeStrategy(llm=llm)


    critique, external_tool_info = strategy.generate_critique(
        idx=0,
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": tests},
        use_tool=True,
        max_interactions=7
    )


def test_create_output_dict() -> None:
    """Tests CriticCodeStrategy create_output_dict."""

def test_update_answer_based_on_critique() -> None:
    """Tests CriticCodeStrategy update_answer_based_on_critique."""

def test_halting_condition() -> None:
    """Tests CriticCodeStrategy halting_condition."""

def test_reset() -> None:
    """Tests CriticCodeStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all code strategies."""