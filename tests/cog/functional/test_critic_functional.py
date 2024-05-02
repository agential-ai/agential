"""Unit tests for CRITIC functional methods."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.critic import (
    _prompt_agent,
    _prompt_critic,
)


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
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


def test__prompt_critic() -> None:
    """Test _prompt_critic function."""
    out = _prompt_critic(
        llm=FakeListChatModel(responses=["1"]), question="", examples="", answer=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_critic(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        prompt="{question}{examples}",
    )
    assert out == "1"
