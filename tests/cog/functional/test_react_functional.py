"""Unit tests for ReAct functional module."""

import tiktoken

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.react import (
    _build_agent_prompt,
    _is_halted,
    _prompt_agent,
)
from agential.cog.prompts.benchmarks.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        max_steps=1,
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
        llm=FakeListChatModel(responses=["1"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        max_steps=1,
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with custom prompt template string.
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert isinstance(out, str)
    assert out == "1"


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
        "{question} {scratchpad} {examples} {max_steps}",
    )
