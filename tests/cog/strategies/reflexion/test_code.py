"""Unit tests for Reflexion Code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.code import (
    ReflexionCoTCodeStrategy,
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActCodeStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
    parse_code_action_cot,
    parse_code_action_react,
)
from agential.cog.prompts.benchmark.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_REACT
)
from agential.cog.prompts.benchmark.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
)
from agential.cog.prompts.agent.reflexion import (
    REFLEXION_COT_INSTRUCTION_MBPP,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_MBPP,
    REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_COT_INSTRUCTION_HUMANEVAL
)

def test_parse_code_action_cot() -> None:
    """Tests parse_code_action_cot."""
    # Test case 1: Correct Finish action.
    action = "Finish```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("Finish", "print('Hello, World!')")

    # Test case 2: No action type.
    action = "```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("", "")

    # Test case 3: Incorrect action type.
    action = "End```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("", "")

    # Test case 4: Finish action with mixed case.
    action = "fIniSh```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("Finish", "print('Hello, World!')")


def test_parse_code_action_react() -> None:
    """Tests parse_code_action_react."""
    # Test case 1: Correct Finish action.
    action = "Finish```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("Finish", "print('Hello, World!')")

    # Test case 2: Correct Implement action.
    action = "Implement```python\nx = 10\n```"
    assert parse_code_action_react(action) == ("Implement", "x = 10")

    # Test case 3: Correct Test action.
    action = "Test```python\nassert x == 10\n```"
    assert parse_code_action_react(action) == ("Test", "assert x == 10")

    # Test case 4: No action type.
    action = "```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("", "")

    # Test case 5: Incorrect action type.
    action = "End```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("", "")

    # Test case 6: Mixed case action types.
    action = "FiNiSh```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("Finish", "print('Hello, World!')")

    action = "imPlEmEnT```python\nx = 10\n```"
    assert parse_code_action_react(action) == ("Implement", "x = 10")

    action = "tEsT```python\nassert x == 10\n```"
    assert parse_code_action_react(action) == ("Test", "assert x == 10")


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTCodeStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTCodeStrategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_out = "Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far to identify the first repeated character."
    gt_scratchpad = "\nThought: Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far to identify the first repeated character."
    responses = [
        "Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far to identify the first repeated character.\nAction: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTCodeStrategy generate_action."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_scratchpad = '\nAction: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]'
    responses = [
        'Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert action_type == "Finish"
    assert query == 'def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None'
    assert strategy._finished == False
    assert strategy._answer == ""
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_cot_generate_action_humaneval() -> None:
    """Tests ReflexionCoTHEvalStrategy generate_action."""
    inst = {"task_id": "HumanEval/0", "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n", "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"}
    question = inst['prompt']
    key = f"{inst['test']}\ncheck({inst['entry_point']})"

    gt_query = '\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n'
    gt_scratchpad = '\nAction: Finish[\n```python\n\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n```\n]'
    responses = [
            'To solve this problem, we need to iterate through the list of numbers and compare the absolute difference between each pair of numbers. If the absolute difference is less than the threshold, we return True. If we finish iterating through the list without finding any close elements, we return False.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTHEvalStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert query == gt_query
    assert strategy._finished == False
    assert strategy._answer == ""
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTCodeStrategy generate_observation."""
    llm = FakeListChatModel(responses=[])

    # Case 1: action_type is "Finish" and answer is correct.
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish",
        query="print('Hello World!')",
        key="print('Hi World!')",
    )
    assert is_correct == True
    assert obs == "Answer is CORRECT"
    assert "Observation: Answer is CORRECT" in strategy._scratchpad
    
    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish",
        query="correct_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in strategy._scratchpad

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Calculate",
        query="some_query",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == 'Invalid action type, please try again. Valid action is Finish[```python<code>```]'
    assert "Observation: Invalid action type, please try again." in strategy._scratchpad


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTCodeStrategy create_output_dict."""
    strategy = ReflexionCoTCodeStrategy(llm=FakeListChatModel(responses=[]))

    # Setting a dummy answer for testing.
    strategy._answer = "correct_answer"

    # Test case 1: Correct answer.
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        obs="Observation: Answer is CORRECT",
        is_correct=True,
        reflections=[],
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "observation": "Observation: Answer is CORRECT",
        "answer": "correct_answer",
        "is_correct": True,
        "reflections": [],
    }
    assert output == expected_output

    # Test case 2: Incorrect answer.
    strategy._answer = "incorrect_answer"
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        obs="Observation: Answer is INCORRECT",
        is_correct=False,
        reflections=[],
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "observation": "Observation: Answer is INCORRECT",
        "answer": "incorrect_answer",
        "is_correct": False,
        "reflections": [],
    }
    assert output == expected_output


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTCodeStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTCodeStrategy(llm=llm, max_trials=3)

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == True

    strategy._answer = "correct_answer"
    assert strategy.halting_condition(2, "correct_answer") == False

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(2, "correct_answer") == False


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTCodeStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTCodeStrategy(llm=llm, max_trials=3)

    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 1: Reset everything.
    strategy.reset()
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""

    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 2: Reset only scratchpad.
    strategy.reset(only_scratchpad=True)
    assert strategy._scratchpad == ""
    assert strategy._finished == True
    assert strategy._answer == "Some answer"


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTCodeStrategy reflect."""


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTCodeStrategy reflect_condition."""


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTCodeStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    humaneval_strategy = ReflexionCoTHEvalStrategy(llm=llm)
    mbpp_strategy = ReflexionCoTMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReflexionCoTHEvalStrategy)
    assert isinstance(mbpp_strategy, ReflexionCoTMBPPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActCodeStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActCodeStrategy generate."""


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActCodeStrategy generate_action."""


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActCodeStrategy generate_observation."""


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActCodeStrategy create_output_dict."""


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActCodeStrategy react_create_output_dict."""


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActCodeStrategy halting_condition."""


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActCodeStrategy react_halting_condition."""


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActCodeStrategy reset."""


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActCodeStrategy reflect."""


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActCodeStrategy reflect_condition."""


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActCodeStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    humaneval_strategy = ReflexionReActHEvalStrategy(llm=llm)
    mbpp_strategy = ReflexionReActMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReflexionReActHEvalStrategy)
    assert isinstance(mbpp_strategy, ReflexionReActMBPPStrategy)
