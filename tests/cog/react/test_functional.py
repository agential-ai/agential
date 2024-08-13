"""Unit tests for ReAct functional module."""

import tiktoken

from litellm.types.utils import ModelResponse

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.react.functional import (
    _build_agent_prompt,
    _is_halted,
    _prompt_agent,
    parse_qa_action,
    parse_math_action,
    parse_code_action
)
from agential.cog.react.prompts import REACT_INSTRUCTION_HOTPOTQA
from agential.llm.llm import MockLLM


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        max_steps=1,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )

    assert isinstance(prompt, str)

    gt_out = "  examples 1"
    out = _build_agent_prompt(
        question="",
        scratchpad="",
        examples="examples",
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert out == gt_out


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        max_steps=1,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "1"

    # Test with custom prompt template string.
    out = _prompt_agent(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert isinstance(out, ModelResponse)
    assert out.choices[0].message.content == "1"


def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Test when finish is true.
    assert _is_halted(
        True,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        100,
        gpt3_5_turbo_enc,
        REACT_INSTRUCTION_HOTPOTQA,
    )

    # Test when idx exceeds max_steps.
    assert _is_halted(
        False,
        11,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        100,
        gpt3_5_turbo_enc,
        REACT_INSTRUCTION_HOTPOTQA,
    )

    # Test when encoded prompt exceeds max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        10,
        gpt3_5_turbo_enc,
        REACT_INSTRUCTION_HOTPOTQA,
    )

    # Test when none of the conditions for halting are met.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        100000,
        gpt3_5_turbo_enc,
        REACT_INSTRUCTION_HOTPOTQA,
    )

    # Test edge case when idx equals max_steps.
    assert _is_halted(
        False,
        10,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        100,
        gpt3_5_turbo_enc,
        REACT_INSTRUCTION_HOTPOTQA,
    )

    # Test edge case when encoded prompt equals max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        1603,
        gpt3_5_turbo_enc,
        REACT_INSTRUCTION_HOTPOTQA,
    )

    # Test with custom prompt template string.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        10,
        1603,
        gpt3_5_turbo_enc,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )


def test_parse_qa_action() -> None:
    """Test parse_qa_action function."""
    # Test with a valid action string.
    valid_string = "ActionType[Argument]"
    assert parse_qa_action(valid_string) == ("ActionType", "Argument")

    # Test with an invalid action string (missing brackets).
    invalid_string = "ActionType Argument"
    assert parse_qa_action(invalid_string) == ("", "")

    # Test with an invalid action string (no action type).
    invalid_string = "[Argument]"
    assert parse_qa_action(invalid_string) == ("", "")

    # Test with an invalid action string (no argument).
    invalid_string = "ActionType[]"
    assert parse_qa_action(invalid_string) == ("", "")


def test_parse_math_action() -> None:
    """Test parse_math_action."""
    test_cases = [
        {
            "input": "Calculate[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Calculate", "def add(a, b): return a + b"),
        },
        {
            "input": "Finish[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Finish", "assert add(2, 3) == 5"),
        },
        {
            "input": "Finish[```python\nThe function is complete.\n```]",
            "expected": ("Finish", "The function is complete."),
        },
        {
            "input": "calculate[```python\ndef subtract(a, b): return a - b\n```]",
            "expected": ("Calculate", "def subtract(a, b): return a - b"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Calculate[```python\nassert subtract(5, 3) == 2\n```]",
            "expected": ("Calculate", "assert subtract(5, 3) == 2"),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
        {
            "input": "Finish[```python\n \n```]",
            "expected": ("Finish", ""),
        },
        {
            "input": "Calculate[```python\nfor i in range(10):\n    print(i)\n```]",
            "expected": ("Calculate", "for i in range(10):\n    print(i)"),
        },
    ]

    for case in test_cases:
        result = parse_math_action(case["input"])
        assert result == case["expected"]


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