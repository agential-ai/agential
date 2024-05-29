"""Unit tests for ReAct Code strategies."""

from tiktoken import Encoding
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.prompts.agents.react import REACT_INSTRUCTION_MBPP
from agential.cog.prompts.benchmarks.mbpp import MBPP_FEWSHOT_EXAMPLES_REACT
from agential.cog.strategies.react.code import (
    ReActCodeStrategy,
    ReActHEvalStrategy,
    ReActMBPPStrategy
)


def test_init() -> None:
    """Test ReActCodeStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 3896
    assert isinstance(strategy.enc, Encoding)
    assert strategy._current_answer == ""
    assert strategy._scratchpad == ""
    assert strategy._finished == False


def test_generate() -> None:
    """Tests ReActCodeStrategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_out = 'I need to find a way to identify the first repeated character in a given string.\n'
    responses = [
        'I need to find a way to identify the first repeated character in a given string.\nAction: Implement[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]\nObservation: The function `first_repeated_char` is implemented to iterate through the string and return the first repeated character encountered.\nThought: I need to test the function to ensure it works correctly with different test cases.\nAction: Test[\n```python\nassert first_repeated_char("abcabc") == "a"\nassert first_repeated_char("abc") == None\nassert first_repeated_char("123123") == "1"\n```\n]\nObservation: All tests passed successfully.\nThought: The function correctly identifies the first repeated character in the given string.\nFinish:[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReActCodeStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
    )
    assert out == gt_out


def test_generate_action() -> None:
    """Tests ReActCodeStrategy generate_action."""
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_query = 'def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None'
    responses = [
        'Implement[\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\n]'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReActCodeStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_MBPP,
        additional_keys={"tests": tests}
    )
    assert action_type == 'Implement'
    assert query == gt_query


def test_generate_observation() -> None:
    """Tests ReActCodeStrategy generate_observation."""

    # Test Implement.
    gt_scratchpad = '\nObservation 0: Done'
    action_type = "Implement"
    query = 'def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None'
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    obs = strategy.generate_observation(
        idx=0,
        action_type=action_type,
        query=query
    )
    assert obs == "Done"
    assert strategy._current_answer == query
    assert strategy._finished is False
    assert strategy._scratchpad == gt_scratchpad

    # Test test.
    gt_scratchpad = '\nObservation 0: Done'
    action_type = "Test"
    query = 'def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None'
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    strategy._current_answer = "print('Hello World')"
    obs = strategy.generate_observation(
        idx=0,
        action_type=action_type,
        query=query
    )
    assert obs == "Done"
    assert strategy._current_answer == "print('Hello World')"
    assert strategy._finished is False
    assert strategy._scratchpad == gt_scratchpad

    # Test finish.
    gt_scratchpad = '\nObservation 0: def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None'
    action_type = "Finish"
    query = 'def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None'
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    obs = strategy.generate_observation(
        idx=0,
        action_type=action_type,
        query=query
    )
    assert obs == query
    assert strategy._current_answer == query
    assert strategy._finished is True
    assert strategy._scratchpad == gt_scratchpad

    # Test error case.


def test_create_output_dict() -> None:

    """Tests ReActCodeStrategy create_output_dict."""

def test_halting_condition() -> None:
    """Tests ReActCodeStrategy halting_condition."""

def test_reset() -> None:
    """Tests ReActCodeStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = FakeListChatModel(responses=[])
    humaneval_strategy = ReActHEvalStrategy(llm=llm)
    mbpp_strategy = ReActMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReActHEvalStrategy)
    assert isinstance(mbpp_strategy, ReActMBPPStrategy)
