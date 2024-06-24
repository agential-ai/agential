"""Unit tests for ReAct Code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from agential.cog.prompts.agent.react import REACT_INSTRUCTION_MBPP
from agential.cog.prompts.benchmark.mbpp import MBPP_FEWSHOT_EXAMPLES_REACT
from agential.cog.strategies.react.code import (
    ReActCodeStrategy,
    ReActHEvalStrategy,
    ReActMBPPStrategy,
    parse_code_action,
)


def test_parse_code_action() -> None:
    """Test parse_code_action."""
    test_cases = [
        {
            "input": "Implement[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Implement", "def add(a, b): return a + b"),
        },
        {
            "input": "Test[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Test", "assert add(2, 3) == 5"),
        },
        {
            "input": "Finish[```python\nThe function is complete.\n```]",
            "expected": ("Finish", "The function is complete."),
        },
        {
            "input": "implement[```python\ndef subtract(a, b): return a - b\n```]",
            "expected": ("Implement", "def subtract(a, b): return a - b"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Test[```python\nassert subtract(5, 3) == 2\n```]",
            "expected": ("Test", "assert subtract(5, 3) == 2"),
        },
    ]

    for case in test_cases:
        result = parse_code_action(case["input"])
        assert result == case["expected"]


def test_init() -> None:
    """Test ReActCodeStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 3896
    assert isinstance(strategy.enc, Encoding)
    assert strategy._answer == ""
    assert strategy._scratchpad == ""
    assert strategy._finished == False


def test_generate() -> None:
    """Tests ReActCodeStrategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    tests = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_out = "I need to find a way to identify the first repeated character in a given string."
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

    gt_query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    responses = [
        "Implement[\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReActCodeStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_MBPP,
        additional_keys={"tests": tests},
    )
    assert action_type == "Implement"
    assert query == gt_query


def test_generate_observation() -> None:
    """Tests ReActCodeStrategy generate_observation."""
    # Test Implement.
    gt_obs = "\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    gt_scratchpad = "\nObservation 0: \n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    action_type = "Implement"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    obs, external_tool_info = strategy.generate_observation(idx=0, action_type=action_type, query=query)
    assert obs == gt_obs
    assert strategy._answer == query
    assert strategy._finished is False
    assert strategy._scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done"}

    # Test test.
    gt_obs = "\n```python\nprint('Hello World')\n\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    gt_scratchpad = "\nObservation 0: \n```python\nprint('Hello World')\n\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```\nExecution Status: Done"
    action_type = "Test"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    strategy._answer = "print('Hello World')"
    obs, external_tool_info = strategy.generate_observation(idx=0, action_type=action_type, query=query)
    assert obs == gt_obs
    assert strategy._answer == "print('Hello World')"
    assert strategy._finished is False
    assert strategy._scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done"}

    # Test finish.
    gt_obs = "\n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```"
    gt_scratchpad = "\nObservation 0: \n```python\ndef first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None\n```"
    action_type = "Finish"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    obs, external_tool_info = strategy.generate_observation(idx=0, action_type=action_type, query=query)
    assert obs == gt_obs
    assert strategy._answer == query
    assert strategy._finished is True
    assert strategy._scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done"}

    # Test error case.
    gt_scratchpad = "\nObservation 0: Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
    action_type = "Unknown"
    query = "def first_repeated_char(s):\n    char_set = set()\n    for char in s:\n        if char in char_set:\n            return char\n        else:\n            char_set.add(char)\n    return None"
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    obs, external_tool_info = strategy.generate_observation(idx=0, action_type=action_type, query=query)
    assert (
        obs
        == "Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
    )
    assert strategy._answer == ""
    assert strategy._finished is False
    assert strategy._scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": ""}


def test_create_output_dict() -> None:
    """Tests ReActCodeStrategy create_output_dict."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    thought = "Sample thought"
    action_type = "implement"
    query = "def add(a, b): return a + b"
    obs = "Execution succeeded"
    strategy._answer = "def add(a, b): return a + b"
    external_tool_info = {"execution_status": "Done"}

    expected_output = {
        "thought": thought,
        "action_type": action_type,
        "query": query,
        "observation": obs,
        "answer": strategy._answer,
        "external_tool_info": external_tool_info
    }

    output = strategy.create_output_dict(thought, action_type, query, obs, external_tool_info)
    assert output == expected_output


def test_halting_condition() -> None:
    """Tests ReActCodeStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    strategy._finished = True
    idx = 5
    question = "What is the sum of 2 and 3?"
    examples = ""
    prompt = "Answer the question."
    additional_keys = {}

    result = strategy.halting_condition(
        idx, question, examples, prompt, additional_keys
    )
    assert result


def test_reset() -> None:
    """Tests ReActCodeStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActCodeStrategy(llm=llm)
    strategy._answer = "def add(a, b): return a + b"
    strategy._scratchpad = "Some scratchpad content"
    strategy._finished = True

    strategy.reset()

    assert strategy._answer == ""
    assert strategy._scratchpad == ""
    assert not strategy._finished


def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = FakeListChatModel(responses=[])
    humaneval_strategy = ReActHEvalStrategy(llm=llm)
    mbpp_strategy = ReActMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReActHEvalStrategy)
    assert isinstance(mbpp_strategy, ReActMBPPStrategy)
