"""Unit tests for ReAct Code strategies."""

from tiktoken import Encoding

from agential.cog.fewshots.mbpp import MBPP_FEWSHOT_EXAMPLES_REACT
from agential.cog.react.prompts import REACT_INSTRUCTION_MBPP
from agential.cog.react.strategies.code import (
    ReActCodeStrategy,
    ReActHEvalStrategy,
    ReActMBPPStrategy,
)
from agential.llm.llm import BaseLLM, MockLLM, Response


def test_init() -> None:
    """Test ReActCodeStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert isinstance(strategy.enc, Encoding)


def test_generate_action() -> None:
    """Tests ReActCodeStrategy generate_action."""
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    gt_scratchpad = "\nAction 0: Implement[\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\n]"
    responses = [
        "Implement[\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActCodeStrategy(llm=llm)
    scratchpad, action_type, query, action_response = strategy.generate_action(
        idx=0,
        scratchpad="",
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
    )
    assert action_type == "Implement"
    assert query == gt_query

    assert scratchpad == gt_scratchpad
    assert action_response == Response(
        input_text="",
        output_text="Implement[\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\n]",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Tests ReActCodeStrategy generate_observation."""
    # Test Implement.
    gt_obs = "\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    gt_scratchpad = "\nObservation 0: \n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    action_type = "Implement"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert obs == gt_obs
    assert answer == query
    assert finished is False
    assert scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done"}
    assert strategy._answer == query

    # Test test.
    gt_obs = "\n```python\n\n\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    gt_scratchpad = "\nObservation 0: \n```python\n\n\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    action_type = "Test"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    answer = "print('Hello World')"
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert obs == gt_obs
    assert answer == ""
    assert finished is False
    assert scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done"}
    assert strategy._answer == ""

    # Test finish.
    gt_obs = "\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```"
    gt_scratchpad = "\nObservation 0: \n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```"
    action_type = "Finish"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert obs == gt_obs
    assert answer == query
    assert finished is True
    assert scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done"}
    assert strategy._answer == query

    # Test error case.
    gt_scratchpad = "\nObservation 0: Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
    action_type = "Unknown"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert (
        obs
        == "Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
    )
    assert answer == ""
    assert finished is False
    assert scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": ""}
    assert strategy._answer == ""


def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    humaneval_strategy = ReActHEvalStrategy(llm=llm)
    mbpp_strategy = ReActMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReActHEvalStrategy)
    assert isinstance(mbpp_strategy, ReActMBPPStrategy)
