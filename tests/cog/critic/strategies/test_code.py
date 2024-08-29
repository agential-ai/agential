"""Unit tests for CRITIC code strategies."""

import pytest

from agential.cog.critic.output import CriticOutput, CriticStepOutput
from agential.cog.critic.prompts import (
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_MBPP,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
)
from agential.cog.critic.strategies.code import (
    CriticCodeStrategy,
    CriticHEvalStrategy,
    CriticMBPPStrategy,
)
from agential.cog.fewshots.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.fewshots.mbpp import MBPP_FEWSHOT_EXAMPLES_POT
from agential.llm.llm import MockLLM, Response


def test_init() -> None:
    """Test CriticCodeStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CriticCodeStrategy(llm=llm)
    assert strategy.llm == llm


def test_generate() -> None:
    """Tests CriticCodeStrategy generate."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"

    use_tool = True

    gt_out = CriticOutput(
        answer="def has_close_elements(numbers, threshold):\n    return any(abs(x - y) < threshold for i, x in enumerate(numbers) for j, y in enumerate(numbers) if i != j)",
        total_prompt_tokens=20,
        total_completion_tokens=40,
        total_tokens=60,
        total_prompt_cost=3e-05,
        total_completion_cost=7.999999999999999e-05,
        total_cost=0.00010999999999999999,
        total_prompt_time=1.0,
        total_time=0.5,
        additional_info=[
            CriticStepOutput(
                answer="def has_close_elements(numbers, threshold):\n    return any(abs(x - y) < threshold for i, x in enumerate(numbers) for j, y in enumerate(numbers) if i != j)",
                critique="The function `has_close_elements` has a correct implementation, utilizing a generator expression with the `any` function to efficiently check if any two numbers in the list are closer to each other than the given threshold. The logic compares all pairs of numbers in the list except for pairs where the indices are the same, ensuring no number is compared with itself.\n\nThere are no issues with the function's design or implementation. The function correctly checks for close elements based on the specified threshold and passes the provided test cases successfully.\n\nTherefore, there are no problems with the given code.",
                external_tool_info={"execution_status": "Done"},
                answer_response=[
                    Response(
                        input_text="",
                        output_text="```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(x - y) < threshold for i, x in enumerate(numbers) for j, y in enumerate(numbers) if i != j)\n```",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
                critique_response=[
                    Response(
                        input_text="",
                        output_text="The function `has_close_elements` has a correct implementation, utilizing a generator expression with the `any` function to efficiently check if any two numbers in the list are closer to each other than the given threshold. The logic compares all pairs of numbers in the list except for pairs where the indices are the same, ensuring no number is compared with itself.\n\nThere are no issues with the function's design or implementation. The function correctly checks for close elements based on the specified threshold and passes the provided test cases successfully.\n\nTherefore, there are no problems with the given code.",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
            )
        ],
    )
    responses = [
        "```python\ndef has_close_elements(numbers, threshold):\n    return any(abs(x - y) < threshold for i, x in enumerate(numbers) for j, y in enumerate(numbers) if i != j)\n```",
        "The function `has_close_elements` has a correct implementation, utilizing a generator expression with the `any` function to efficiently check if any two numbers in the list are closer to each other than the given threshold. The logic compares all pairs of numbers in the list except for pairs where the indices are the same, ensuring no number is compared with itself.\n\nThere are no issues with the function's design or implementation. The function correctly checks for close elements based on the specified threshold and passes the provided test cases successfully.\n\nTherefore, there are no problems with the given code.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strat = CriticHEvalStrategy(llm=llm, testing=True)
    out = strat.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_HUMANEVAL,
        critique_examples=(
            HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC
            if use_tool
            else HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL
        ),
        critique_prompt=(
            CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL
            if use_tool
            else CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL
        ),
        additional_keys={},
        critique_additional_keys={"tests": tests},
        max_interactions=3,
        use_tool=use_tool,
        reset=True,
    )
    assert out == gt_out

    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""
    use_tool = True

    gt_out = CriticOutput(
        answer='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
        total_prompt_tokens=60,
        total_completion_tokens=120,
        total_tokens=180,
        total_prompt_cost=9e-05,
        total_completion_cost=0.00023999999999999998,
        total_cost=0.00033,
        total_prompt_time=3.0,
        total_time=0.5,
        additional_info=[
            CriticStepOutput(
                answer='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                critique="There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
                external_tool_info={
                    "execution_status": "IndentationError('unexpected indent', ('<string>', 15, 4, '    assert first_repeated_char(\"abc\") == None\\n', 15, -1))"
                },
                answer_response=[
                    Response(
                        input_text="",
                        output_text='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
                critique_response=[
                    Response(
                        input_text="",
                        output_text="There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
            ),
            CriticStepOutput(
                answer='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                critique="There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
                external_tool_info={
                    "execution_status": "IndentationError('unexpected indent', ('<string>', 15, 4, '    assert first_repeated_char(\"abc\") == None\\n', 15, -1))"
                },
                answer_response=[
                    Response(
                        input_text="",
                        output_text='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
                critique_response=[
                    Response(
                        input_text="",
                        output_text="There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
            ),
            CriticStepOutput(
                answer='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                critique="There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
                external_tool_info={
                    "execution_status": "IndentationError('unexpected indent', ('<string>', 15, 4, '    assert first_repeated_char(\"abc\") == None\\n', 15, -1))"
                },
                answer_response=[
                    Response(
                        input_text="",
                        output_text='def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
                critique_response=[
                    Response(
                        input_text="",
                        output_text="There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    )
                ],
            ),
        ],
    )
    responses = [
        'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Run the tests\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"',
        "There is no problem with the code provided. The function correctly finds the first repeated character in a given string by using a set to keep track of characters already seen. It returns the first character that appears more than once, or None if there are no repeated characters. The function passes the provided tests successfully.",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strat = CriticMBPPStrategy(llm=llm, testing=True)

    out = strat.generate(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_MBPP,
        critique_examples=(
            MBPP_FEWSHOT_EXAMPLES_CRITIC
            if use_tool
            else MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL
        ),
        critique_prompt=(
            CRITIC_CRITIQUE_INSTRUCTION_MBPP
            if use_tool
            else CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP
        ),
        additional_keys={"tests": tests},
        critique_additional_keys={"tests": tests},
        max_interactions=3,
        use_tool=use_tool,
        reset=True,
    )
    assert out == gt_out


def test_generate_answer() -> None:
    """Tests CriticCodeStrategy generate_answer."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]

    gt_result = "    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"
    responses = [
        "```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    result, answer_response = strategy.generate_answer(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert result == gt_result
    assert answer_response == [
        Response(
            input_text="",
            output_text="```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]


def test_generate_critique() -> None:
    """Tests CriticCodeStrategy generate_critique."""
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
assert first_repeated_char("abc") == None
assert first_repeated_char("123123") == "1\""""

    # Test with no tool.
    gt_critique = "There is no problem with the code provided. The function `first_repeated_char` correctly iterates over the characters in the string and keeps track of seen characters using a set. If a character is already in the set, it returns that character as the first repeated character. Otherwise, it adds the character to the set and continues. The function passes the given test cases without any issues."
    responses = [
        "There is no problem with the code provided. The function `first_repeated_char` correctly iterates over the characters in the string and keeps track of seen characters using a set. If a character is already in the set, it returns that character as the first repeated character. Otherwise, it adds the character to the set and continues. The function passes the given test cases without any issues."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    answer = 'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"'
    critique, external_tool_info, finished, critique_response = (
        strategy.generate_critique(
            idx=0,
            question=question,
            examples=MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
            answer=answer,
            critique="",
            prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
            additional_keys={"tests": tests},
            use_tool=False,
            max_interactions=7,
        )
    )

    assert critique == gt_critique
    assert external_tool_info == {"execution_status": ""}
    assert not finished
    assert critique_response == [
        Response(
            input_text="",
            output_text="There is no problem with the code provided. The function `first_repeated_char` correctly iterates over the characters in the string and keeps track of seen characters using a set. If a character is already in the set, it returns that character as the first repeated character. Otherwise, it adds the character to the set and continues. The function passes the given test cases without any issues.",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]

    # Test no tests error.
    with pytest.raises(ValueError):
        critique, external_tool_info, finished, critique_response = (
            strategy.generate_critique(
                idx=0,
                question=question,
                examples=MBPP_FEWSHOT_EXAMPLES_CRITIC,
                answer=answer,
                critique="",
                prompt=CRITIC_CRITIQUE_INSTRUCTION_MBPP,
                additional_keys={},
                use_tool=True,
                max_interactions=7,
            )
        )

    # Test with tool.
    gt_critique = "There doesn't seem to be any issue with the provided code for finding the first repeated character in a given string. The function correctly uses a set to keep track of seen characters and returns the first repeated character encountered.\n\nThe function passes the provided test cases and seems to be implemented correctly."
    responses = [
        "There doesn't seem to be any issue with the provided code for finding the first repeated character in a given string. The function correctly uses a set to keep track of seen characters and returns the first repeated character encountered.\n\nThe function passes the provided test cases and seems to be implemented correctly."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    answer = 'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"'
    critique, external_tool_info, finished, critique_response = (
        strategy.generate_critique(
            idx=0,
            question=question,
            examples=MBPP_FEWSHOT_EXAMPLES_CRITIC,
            answer=answer,
            critique="",
            prompt=CRITIC_CRITIQUE_INSTRUCTION_MBPP,
            additional_keys={"tests": tests},
            use_tool=True,
            max_interactions=7,
        )
    )

    assert critique == gt_critique
    assert external_tool_info == {"execution_status": "Done"}
    assert finished
    assert critique_response == [
        Response(
            input_text="",
            output_text="There doesn't seem to be any issue with the provided code for finding the first repeated character in a given string. The function correctly uses a set to keep track of seen characters and returns the first repeated character encountered.\n\nThe function passes the provided test cases and seems to be implemented correctly.",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]


def test_create_output_dict() -> None:
    """Tests CriticCodeStrategy create_output_dict."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CriticCodeStrategy(llm=llm)
    result = strategy.create_output_dict(
        finished=True,
        answer="",
        critique="",
        external_tool_info={"a": "b"},
        answer_response=[],
        critique_response=[],
    )
    assert result == {
        "answer": "",
        "critique": "",
        "external_tool_info": {"a": "b"},
        "critique_response": [],
        "answer_response": [],
    }


def test_update_answer_based_on_critique() -> None:
    """Tests CriticCodeStrategy update_answer_based_on_critique."""
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
assert first_repeated_char("abc") == None
assert first_repeated_char("123123") == "1\""""

    gt_new_answer = "The provided code for finding the first repeated character in a given string is correct and passes the test cases. No issues were identified with the implementation."
    critique = "There doesn't seem to be any issue with the provided code for finding the first repeated character in a given string. The function correctly uses a set to keep track of seen characters and returns the first repeated character encountered.\n\nThe function passes the provided test cases and seems to be implemented correctly."
    answer = 'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n\n# Testing the function with the given test cases\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"'
    responses = [
        "The provided code for finding the first repeated character in a given string is correct and passes the test cases. No issues were identified with the implementation."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticCodeStrategy(llm=llm)
    new_answer, answer_response = strategy.update_answer_based_on_critique(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique=critique,
        prompt=CRITIC_CRITIQUE_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
        external_tool_info={"execution_status": "Done"},
    )

    assert new_answer == gt_new_answer
    assert answer_response == [
        Response(
            input_text="",
            output_text="The provided code for finding the first repeated character in a given string is correct and passes the test cases. No issues were identified with the implementation.",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]


def test_halting_condition() -> None:
    """Tests CriticCodeStrategy halting_condition."""
    strategy = CriticCodeStrategy(llm=None)

    assert strategy.halting_condition(False) is False

    assert strategy.halting_condition(True) is True


def test_reset() -> None:
    """Tests CriticCodeStrategy reset."""
    strategy = CriticCodeStrategy(llm=None)
    strategy.reset()


def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    heval_strategy = CriticHEvalStrategy(llm=llm)
    mbpp_strategy = CriticMBPPStrategy(llm=llm)

    assert isinstance(heval_strategy, CriticHEvalStrategy)
    assert isinstance(mbpp_strategy, CriticMBPPStrategy)


def test_heval_generate_critique() -> None:
    """Tests CriticHEvalStrategy generate_critique."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]
    answer = "    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"

    # Test no tool.
    gt_critique = "The implementation of the `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. However, there is a minor issue with the threshold comparison logic.\n\nIn the comparison `abs(numbers[i] - numbers[j]) < threshold`, the condition is checking if the absolute difference between two numbers is less than the threshold. This condition is correct for identifying close elements. However, the problem arises when the difference between two numbers is exactly equal to the threshold, as the function is expected to return False in that case.\n\nFor example, if the list is `[1.0, 2.0, 3.0]` and the threshold is `1.0`, the function should return False because none of the numbers have a difference exactly equal to the threshold. However, the current implementation would return True because the condition allows for numbers with a difference less than the threshold.\n\nTo fix this issue and align the function with the expected behavior, the threshold comparison should be modified to `abs(numbers[i] - numbers[j]) <= threshold` to include the case where the difference is exactly equal to the threshold."
    responses = [
        'The implementation of the `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. However, there is a minor issue with the threshold comparison logic.\n\nIn the comparison `abs(numbers[i] - numbers[j]) < threshold`, the condition is checking if the absolute difference between two numbers is less than the threshold. This condition is correct for identifying close elements. However, the problem arises when the difference between two numbers is exactly equal to the threshold, as the function is expected to return False in that case.\n\nFor example, if the list is `[1.0, 2.0, 3.0]` and the threshold is `1.0`, the function should return False because none of the numbers have a difference exactly equal to the threshold. However, the current implementation would return True because the condition allows for numbers with a difference less than the threshold.\n\nTo fix this issue and align the function with the expected behavior, the threshold comparison should be modified to `abs(numbers[i] - numbers[j]) <= threshold` to include the case where the difference is exactly equal to the threshold.\n\nHere\'s the corrected implementation of the `has_close_elements` function:\n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) <= threshold:\n                return True\n    return False\n```\n\nWith this modification, the function will now correctly handle cases where the difference between two numbers is exactly equal to the threshold.'
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticHEvalStrategy(llm=llm)

    critique, external_tool_info, finished, critique_response = (
        strategy.generate_critique(
            idx=0,
            question=question,
            examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
            answer=answer,
            critique="",
            prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
            additional_keys={"tests": tests},
            use_tool=False,
            max_interactions=7,
        )
    )

    assert critique == gt_critique
    assert external_tool_info == {}
    assert not finished
    assert critique_response == [
        Response(
            input_text="",
            output_text='The implementation of the `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. However, there is a minor issue with the threshold comparison logic.\n\nIn the comparison `abs(numbers[i] - numbers[j]) < threshold`, the condition is checking if the absolute difference between two numbers is less than the threshold. This condition is correct for identifying close elements. However, the problem arises when the difference between two numbers is exactly equal to the threshold, as the function is expected to return False in that case.\n\nFor example, if the list is `[1.0, 2.0, 3.0]` and the threshold is `1.0`, the function should return False because none of the numbers have a difference exactly equal to the threshold. However, the current implementation would return True because the condition allows for numbers with a difference less than the threshold.\n\nTo fix this issue and align the function with the expected behavior, the threshold comparison should be modified to `abs(numbers[i] - numbers[j]) <= threshold` to include the case where the difference is exactly equal to the threshold.\n\nHere\'s the corrected implementation of the `has_close_elements` function:\n\n```python\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) <= threshold:\n                return True\n    return False\n```\n\nWith this modification, the function will now correctly handle cases where the difference between two numbers is exactly equal to the threshold.',
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]

    # Test no tests error.
    with pytest.raises(ValueError):
        critique, external_tool_info, finished, critique_response = (
            strategy.generate_critique(
                idx=0,
                question=question,
                examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
                answer=answer,
                critique="",
                prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
                additional_keys={},
                use_tool=True,
                max_interactions=7,
            )
        )

    # Test with tool.
    gt_critique = "There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation."
    responses = [
        "There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticHEvalStrategy(llm=llm)

    critique, external_tool_info, finished, critique_response = (
        strategy.generate_critique(
            idx=0,
            question=question,
            examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
            answer=answer,
            critique="",
            prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
            additional_keys={"tests": tests},
            use_tool=True,
            max_interactions=7,
        )
    )
    assert critique == gt_critique
    assert external_tool_info == {"execution_status": "Done"}
    assert finished
    assert critique_response == [
        Response(
            input_text="",
            output_text="There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation.",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]


def test_heval_update_answer_based_on_critique() -> None:
    """Tests CriticHEvalStrategy update_answer_based_on_critique."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]
    answer = "    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"
    tests = f"{inst['test']}\ncheck({inst['entry_point']})"

    gt_new_answer = "    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False"
    responses = [
        "```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```"
    ]
    critique = "There is no problem with the provided code. The `has_close_elements` function correctly checks if there are any two numbers in the list that are closer to each other than the given threshold. The function uses a nested loop to compare all pairs of numbers in the list and returns `True` if it finds any pair that meets the condition. The test cases provided in the `check` function also cover a variety of scenarios to verify the correctness of the implementation."
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CriticHEvalStrategy(llm=llm)
    new_answer, answer_response = strategy.update_answer_based_on_critique(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique=critique,
        prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        additional_keys={"tests": tests},
        external_tool_info={"execution_status": "Done"},
    )
    assert new_answer == gt_new_answer
    assert answer_response == [
        Response(
            input_text="",
            output_text="```python\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        )
    ]
