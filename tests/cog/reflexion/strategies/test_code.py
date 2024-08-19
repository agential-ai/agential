"""Unit tests for Reflexion Code strategies."""

from agential.cog.fewshots.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
)
from agential.cog.fewshots.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.reflexion.output import ReflexionCoTOutput, ReflexionCoTStepOutput
from agential.cog.reflexion.prompts import (
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_HUMANEVAL,
    REFLEXION_COT_INSTRUCTION_MBPP,
    REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_REACT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
)
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.reflexion.strategies.code import (
    ReflexionCoTCodeStrategy,
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActCodeStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
)
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.metrics import PromptMetrics


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTCodeStrategy init."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTCodeStrategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_out = ReflexionCoTOutput(
        answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
        total_prompt_tokens=80,
        total_completion_tokens=160,
        total_tokens=240,
        total_prompt_cost=0.00012000000000000002,
        total_completion_cost=0.00031999999999999997,
        total_cost=0.00043999999999999996,
        total_prompt_time=4.0,
        total_time=0.5,
        additional_info=[
            ReflexionCoTStepOutput(
                thought="Let's think step by step. We need to iterate through the characters in the string and keep track of the characters we have seen so far to find the first repeated character.",
                action_type="Finish",
                observation="Answer is INCORRECT",
                answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                is_correct=False,
                reflections=[],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=None,
            ),
            ReflexionCoTStepOutput(
                thought="Finish[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]",
                action_type="Finish",
                observation="Answer is INCORRECT",
                answer="def first_repeated_char(input_str):\n    seen_chars = set()\n    for char in input_str:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None",
                is_correct=False,
                reflections=[
                    "Let's think step by step. We need to iterate through the characters in the string and keep track of the characters we have seen so far to find the first repeated character.Action: Finish[```pythondef first_repeated_char(input_str):    seen_chars = set()    for char in input_str:        if char in seen_chars:            return char        seen_chars.add(char)    return None```]"
                ],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            ReflexionCoTStepOutput(
                thought="Let's think step by step. We need to iterate through the characters in the string and keep track of the characters we have seen so far to find the first repeated character.",
                action_type="Finish",
                observation="Answer is INCORRECT",
                answer="def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None",
                is_correct=False,
                reflections=[
                    "Let's think step by step. We need to iterate through the characters in the string and keep track of the characters we have seen so far to find the first repeated character.Action: Finish[```pythondef first_repeated_char(input_str):    seen_chars = set()    for char in input_str:        if char in seen_chars:            return char        seen_chars.add(char)    return None```]",
                    "Finish[```pythondef first_repeated_char(s):    seen = set()    for char in s:        if char in seen:            return char        seen.add(char)    return None```]",
                ],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=PromptMetrics(
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
        "Let's think step by step. We need to iterate through the characters in the string and keep track of the characters we have seen so far to find the first repeated character.\nAction: Finish[\n```python\ndef first_repeated_char(input_str):\n    seen_chars = set()\n    for char in input_str:\n        if char in seen_chars:\n            return char\n        seen_chars.add(char)\n    return None\n```\n]",
        "Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTCodeStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        key=key,
        examples=MBPP_FEWSHOT_EXAMPLES_COT,
        prompt=REFLEXION_COT_INSTRUCTION_MBPP,
        reflect_examples=MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        reflect_prompt=REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
        reflect_strategy="reflexion",
        additional_keys={"tests": key},
        reflect_additional_keys={"tests": key},
        patience=3,
        reset=True,
    )
    assert out == gt_out


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTCodeStrategy generate_action."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    responses = [
        "Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    scratchpad, action_type, query, action_metrics = strategy.generate_action(
        scratchpad="",
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert action_type == "Finish"
    assert (
        query
        == "def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None"
    )
    print(repr(scratchpad))
    print(repr(action_metrics))
    assert (
        scratchpad
        == "\nAction:  Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    )
    assert action_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_cot_generate_action_humaneval() -> None:
    """Tests ReflexionCoTHEvalStrategy generate_action."""
    inst = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    question = inst["prompt"]
    key = f"{inst['test']}\ncheck({inst['entry_point']})"

    gt_query = "\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n"
    gt_scratchpad = "\nAction: Finish[\n```python\n\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n```\n]"
    responses = [
        "To solve this problem, we need to iterate through the list of numbers and compare the absolute difference between each pair of numbers. If the absolute difference is less than the threshold, we return True. If we finish iterating through the list without finding any close elements, we return False.\n\n```python\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTHEvalStrategy(llm=llm)
    scratchpad, action_type, query, action_metrics = strategy.generate_action(
        scratchpad="",
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert query == gt_query
    assert (
        scratchpad
        == "\nAction: Finish[\n```python\n\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n\n```\n]"
    )
    assert action_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTCodeStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # Case 1: action_type is "Finish" and answer is correct.
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Finish",
        query="print('Hello World!')",
        key="print('Hi World!')",
    )
    assert is_correct == True
    assert obs == "Answer is CORRECT"
    assert "Observation: Answer is CORRECT" in scratchpad
    assert answer == "print('Hello World!')"

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Finish",
        query="correct_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in scratchpad
    assert answer == "correct_answer"


    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Calculate",
        query="some_query",
        key="correct_answer",
    )
    assert is_correct == False
    assert (
        obs
        == "Invalid action type, please try again. Valid action is Finish[```python<code>```]"
    )
    assert "Observation: Invalid action type, please try again." in scratchpad
    assert answer == ""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTCodeStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTCodeStrategy(llm=llm, max_trials=3)

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer", "correct_answer") == True

    strategy._answer = "correct_answer"
    assert strategy.halting_condition(2, "correct_answer", "correct_answer") == False

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(2, "correct_answer", "correct_answer") == False


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTCodeStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTCodeStrategy(llm)

    assert not strategy.reflect_condition(0, "strategy1", "key1", "key2")
    assert strategy.reflect_condition(1, "strategy1", "key1", "key2")
    assert strategy.reflect_condition(1, "strategy1", "key2", "key2")
    assert strategy.reflect_condition(1, "", "key2", "key2")


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTCodeStrategy instantiate strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    humaneval_strategy = ReflexionCoTHEvalStrategy(llm=llm)
    mbpp_strategy = ReflexionCoTMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReflexionCoTHEvalStrategy)
    assert isinstance(mbpp_strategy, ReflexionCoTMBPPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActCodeStrategy init."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}
    assert strategy._prompt_metrics == {"reflection": None}


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActCodeStrategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_scratchpad = "\nThought: Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far. Once we encounter a character that has already been seen, we return it as the first repeated character."
    gt_out = "Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far. Once we encounter a character that has already been seen, we return it as the first repeated character."
    responses = [
        "Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far. Once we encounter a character that has already been seen, we return it as the first repeated character.\nAction: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActCodeStrategy(llm=llm)
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
    assert strategy._prompt_metrics == {"reflection": None}
    assert strategy._prompt_metrics_react == {
        "thought": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
        "action": None,
    }


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActCodeStrategy generate_action."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_scratchpad = "\nAction: Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    gt_query = "def first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None"
    responses = [
        "Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActCodeStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert action_type == "Implement"
    assert query == gt_query
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""
    assert strategy._prompt_metrics == {"reflection": None}
    assert strategy._prompt_metrics_react == {
        "thought": None,
        "action": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
    }


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActCodeStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActCodeStrategy(llm=llm)

    # Test Implement.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Implement",
        query="x = 1 + 1\nanswer = x",
        key="key1",
    )
    assert not is_correct
    assert obs == "\n```python\nx = 1 + 1\nanswer = x\n```\nExecution Status: "
    assert external_tool_info == {"execution_status": "Done"}

    # Test Finish incorrect.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=2,
        action_type="Finish",
        query="answer = 5",
        key="key2",
    )
    assert not is_correct
    assert obs == "Answer is INCORRECT"
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "answer = 5"
    assert external_tool_info == {
        "execution_status": "NameError(\"name 'key2' is not defined\")"
    }

    # Test Finish correct.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=3,
        action_type="Finish",
        query="answer = 5",
        key="print('Hello world')",
    )
    assert is_correct
    assert obs == "Answer is CORRECT"
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "answer = 5"
    assert external_tool_info == {"execution_status": "Done"}

    # Test Test action.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=4,
        action_type="Test",
        query="assert answer == 5",
        key="key4",
    )
    assert is_correct
    assert (
        obs
        == "\n```python\nanswer = 5\n\nassert answer == 5\n```\nExecution Status: Done"
    )
    assert external_tool_info == {"execution_status": "Done"}

    # Test invalid action.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=5,
        action_type="Invalid",
        query="answer = 5",
        key="key5",
    )
    assert not is_correct
    assert (
        obs
        == "Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
    )
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "answer = 5"
    assert external_tool_info == {"execution_status": ""}


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActCodeStrategy create_output_dict."""
    strategy = ReflexionReActCodeStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    react_out = [
        {
            "thought": "First thought",
            "action_type": "Query",
            "query": "What is the capital of France?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
        }
    ]
    reflections = "Reflection on the first thought."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
        "prompt_metrics": {"reflection": None},
    }
    assert output == expected_output


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActCodeStrategy react_create_output_dict."""
    strategy = ReflexionReActCodeStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    # Test case 1: Valid output creation
    output = strategy.react_create_output_dict(
        thought="Initial thought",
        action_type="Query",
        query="What is the capital of France?",
        obs="Observation: Answer is CORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=True,
    )
    expected_output = {
        "thought": "Initial thought",
        "action_type": "Query",
        "query": "What is the capital of France?",
        "observation": "Observation: Answer is CORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": True,
        "prompt_metrics": {"thought": None, "action": None},
    }
    assert output == expected_output


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActCodeStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActCodeStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActCodeStrategy(llm=llm, max_trials=5)
    strategy._answer = "correct_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActCodeStrategy(llm=llm, max_trials=3)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer") == True

    # Test case 4: Halting condition met using max_trials from kwargs.
    strategy = ReflexionReActCodeStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer", max_trials=4) == False

    # Test case 5: Halting condition not met using max_trials from kwargs.
    strategy = ReflexionReActCodeStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer", max_trials=3) == True


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActCodeStrategy react_halting_condition."""
    strategy = ReflexionReActCodeStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    idx = 0
    question = "What is the capital of France?"
    examples = ""
    reflections = ""
    prompt = "Answer the question."

    assert not strategy.react_halting_condition(
        idx, question, examples, reflections, prompt, {}
    )


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActCodeStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActCodeStrategy(llm=llm)
    strategy._scratchpad = "Some previous state"
    strategy._finished = True

    strategy.reset()

    assert strategy._scratchpad == ""
    assert not strategy._finished
    assert strategy._prompt_metrics == {"reflection": None}
    assert strategy._prompt_metrics_react == {"action": None, "thought": None}


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActCodeStrategy reflect."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_reflections = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActCodeStrategy(llm=llm)
    _, reflections = strategy.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert reflections == gt_reflections
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}
    assert strategy._prompt_metrics == {
        "reflection": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        }
    }


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActCodeStrategy reflect_condition."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActCodeStrategy(llm=llm)
    out = strategy.reflect_condition(
        step_idx=1,
        reflect_strategy="reflexion",
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        key="key",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert not out


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActCodeStrategy instantiate strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    humaneval_strategy = ReflexionReActHEvalStrategy(llm=llm)
    mbpp_strategy = ReflexionReActMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReflexionReActHEvalStrategy)
    assert isinstance(mbpp_strategy, ReflexionReActMBPPStrategy)
