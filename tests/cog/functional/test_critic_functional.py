"""Unit tests for CRITIC functional methods."""
import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.functional.critic import (
    _build_agent_prompt,
    _build_critique_prompt,
    _prompt_agent,
    _prompt_critique,
)


@pytest.mark.parametrize("benchmark", ["hotpotqa", "triviaqa"])
def test__build_agent_prompt(benchmark) -> None:
    """Test _build_agent_prompt function."""
    gt_out = "\n(END OF EXAMPLES)\n\nQ: \nA: "

    prompt = _build_agent_prompt(question="", examples="", benchmark=benchmark)
    assert prompt == gt_out, f"Failed for benchmark: {benchmark}"


@pytest.mark.parametrize("benchmark", ["hotpotqa", "triviaqa"])
def test__prompt_agent(benchmark) -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        benchmark=benchmark,
    )
    assert out == "1"


@pytest.mark.parametrize("benchmark", ["hotpotqa", "triviaqa"])
def test__build_critique_prompt(benchmark) -> None:
    """Test _build_critique_prompt function."""
    gt_out = "\n(END OF EXAMPLES)\n\nQuestion: \nProposed Answer: \n\nWhat's the problem with the above answer?\n\n1. Plausibility:\n\n"
    prompt = _build_critique_prompt(
        question="", examples="", answer=""
    )
    assert prompt == gt_out


@pytest.mark.parametrize("benchmark", ["hotpotqa", "triviaqa"])
def test__prompt_critique(benchmark) -> None:
    """Test _prompt_critique function."""
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        benchmark=benchmark,
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        benchmark=benchmark,
    )
    assert out == "1"
