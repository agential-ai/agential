"""Unit tests for ReAct functional module."""
import tiktoken

from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.functional.react import (
    _build_agent_prompt,
    _is_halted,
    _prompt_agent,
)


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(question="", scratchpad="")
    assert isinstance(prompt, str)


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]), question="", scratchpad=""
    )
    assert isinstance(out, str)
    assert out == "1"


def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    assert _is_halted(True, 1, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test when step_n exceeds max_steps.
    assert _is_halted(False, 11, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test when encoded prompt exceeds max_tokens.
    assert _is_halted(False, 1, 10, "question", "scratchpad", 10, gpt3_5_turbo_enc)

    # Test when none of the conditions for halting are met.
    assert _is_halted(False, 1, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test edge case when step_n equals max_steps.
    assert _is_halted(False, 10, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test edge case when encoded prompt equals max_tokens.
    assert _is_halted(False, 1, 10, "question", "scratchpad", 20, gpt3_5_turbo_enc)
