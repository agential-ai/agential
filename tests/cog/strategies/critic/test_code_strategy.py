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
    
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
assert first_repeated_char("abc") == None
assert first_repeated_char("123123") == "1\""""

    # Test with no tool.
    gt_critique = 'There is no problem with the code provided. The function `first_repeated_char` correctly iterates over the characters in the string and keeps track of seen characters using a set. If a character is already in the set, it returns that character as the first repeated character. Otherwise, it adds the character to the set and continues. The function passes the given test cases without any issues.'
    responses = [
        'There is no problem with the code provided. The function `first_repeated_char` correctly iterates over the characters in the string and keeps track of seen characters using a set. If a character is already in the set, it returns that character as the first repeated character. Otherwise, it adds the character to the set and continues. The function passes the given test cases without any issues.'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    answer = 'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"'
    critique, external_tool_info = strategy.generate_critique(
        idx=0,
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
        use_tool=False,
        max_interactions=7
    )

    assert critique == gt_critique
    assert external_tool_info == {}

    # Test no tests error.
    with pytest.raises(ValueError):
        critique, external_tool_info = strategy.generate_critique(
            idx=0,
            question=question,
            examples=MBPP_FEWSHOT_EXAMPLES_CRITIC,
            answer=answer,
            critique="",
            prompt=CRITIC_CRITIQUE_INSTRUCTION_MBPP,
            additional_keys={},
            use_tool=True,
            max_interactions=7
        )

    # Test with tool.
    gt_critique = "There doesn't seem to be any issue with the provided code for finding the first repeated character in a given string. The function correctly uses a set to keep track of seen characters and returns the first repeated character encountered.\n\nThe function passes the provided test cases and seems to be implemented correctly."
    responses = [
        "There doesn't seem to be any issue with the provided code for finding the first repeated character in a given string. The function correctly uses a set to keep track of seen characters and returns the first repeated character encountered.\n\nThe function passes the provided test cases and seems to be implemented correctly."
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    answer = 'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"'
    critique, external_tool_info = strategy.generate_critique(
        idx=0,
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
        use_tool=True,
        max_interactions=7
    )

    assert critique == gt_critique
    assert external_tool_info == {'execution_status': 'Done'}
    assert strategy._halt


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


def test_heval_generate_critique() -> None:
    """Tests CritHEvalCodeStrategy generate_critique."""
    inst = {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}
    question = inst['prompt']
    answer = '    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"

    # Test no tool.
    gt_critique = 'The implementation of the `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. However, there is a minor issue with the threshold comparison logic.\n\nIn the comparison `abs(numbers[i] - numbers[j]) < threshold`, the condition is checking if the absolute difference between two numbers is less than the threshold. This condition is correct for identifying close elements. However, the problem arises when the difference between two numbers is exactly equal to the threshold, as the function is expected to return False in that case.\n\nFor example, if the list is `[1.0, 2.0, 3.0]` and the threshold is `1.0`, the function should return False because none of the numbers have a difference exactly equal to the threshold. However, the current implementation would return True because the condition allows for numbers with a difference less than the threshold.\n\nTo fix this issue and align the function with the expected behavior, the threshold comparison should be modified to `abs(numbers[i] - numbers[j]) <= threshold` to include the case where the difference is exactly equal to the threshold.'
    responses = [
        'The implementation of the `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. However, there is a minor issue with the threshold comparison logic.\n\nIn the comparison `abs(numbers[i] - numbers[j]) < threshold`, the condition is checking if the absolute difference between two numbers is less than the threshold. This condition is correct for identifying close elements. However, the problem arises when the difference between two numbers is exactly equal to the threshold, as the function is expected to return False in that case.\n\nFor example, if the list is `[1.0, 2.0, 3.0]` and the threshold is `1.0`, the function should return False because none of the numbers have a difference exactly equal to the threshold. However, the current implementation would return True because the condition allows for numbers with a difference less than the threshold.\n\nTo fix this issue and align the function with the expected behavior, the threshold comparison should be modified to `abs(numbers[i] - numbers[j]) <= threshold` to include the case where the difference is exactly equal to the threshold.\n\nHere\'s the corrected implementation of the `has_close_elements` function:\n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) <= threshold:\n                return True\n    return False\n```\n\nWith this modification, the function will now correctly handle cases where the difference between two numbers is exactly equal to the threshold.'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CritHEvalCodeStrategy(llm=llm)

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
    gt_critique = 'There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation.'
    responses = [
        'There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation.'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CritHEvalCodeStrategy(llm=llm)

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
    assert critique == gt_critique
    assert external_tool_info == {'execution_status': 'Done'}
    assert strategy._halt


def test_heval_update_answer_based_on_critique() -> None:
    """Tests CritHEvalCodeStrategy update_answer_based_on_critique."""

    inst = {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}
    question = inst['prompt']
    answer = '    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"
    
    gt_critique = '    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    gt_new_answer = '    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    responses = [
        '```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```'
    ]
    critique = 'There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation.'
    llm = FakeListChatModel(responses=responses)
    strategy = CritHEvalCodeStrategy(llm=llm)
    new_answer = strategy.update_answer_based_on_critique(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique=critique,
        prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": tests},
        external_tool_info={"execution_status": "Done"}
    )
    assert new_answer == gt_new_answer
    assert not strategy._halt


def test_heval_reset() -> None:
    """Tests CritHEvalCodeStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = CritHEvalCodeStrategy(llm=llm)
    assert not strategy._halt
    strategy._halt = True
    assert strategy._halt
