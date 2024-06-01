"""Unit tests for CRITIC functional methods."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.critic import (
    _build_agent_prompt,
    _build_critique_prompt,
    _prompt_agent,
    _prompt_critique,
    remove_comment,
)
from agential.cog.prompts.agent.critic import CRITIC_INSTRUCTION_HOTPOTQA, CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA


# Ref: https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/tools/interpreter_api.py.
def test_remove_comments() -> None:
    """Test remove_comments function."""
    code = """# This is a comment\n# Another comment\nint x = 1"""
    expected = "int x = 1"
    result = remove_comment(code)
    assert result == expected


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    gt_out = "\n(END OF EXAMPLES)\n\nQ: \nA: "
    prompt = _build_agent_prompt(
        question="",
        examples="",
        prompt=CRITIC_INSTRUCTION_HOTPOTQA
    )
    assert prompt == gt_out

    # Test custom prompt.
    prompt = _build_agent_prompt(
        question="", examples="", prompt="{question}{examples}"
    )
    assert prompt == ""


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        prompt=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        prompt="{question}{examples}",
    )
    assert out == "1"


def test__build_critique_prompt() -> None:
    """Test _build_critique_prompt function."""
    gt_out = "\n(END OF EXAMPLES)\n\nQuestion: \nProposed Answer: \n\nWhat's the problem with the above answer?\n\n1. Plausibility:\n\n"
    prompt = _build_critique_prompt(question="", examples="", answer="", critique="", prompt=CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA)
    assert prompt == gt_out

    # Test custom prompt.
    prompt = _build_critique_prompt(
        question="",
        examples="",
        answer="",
        critique="",
        prompt="{question}{examples}{answer}{critique}",
    )
    assert prompt == ""


def test__prompt_critique() -> None:
    """Test _prompt_critique function."""
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]), question="", examples="", answer="", critique="", prompt=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        critique="",
        prompt="{question}{examples}{critique}",
    )
    assert out == "1"
