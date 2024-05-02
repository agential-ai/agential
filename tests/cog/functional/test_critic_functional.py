"""Unit tests for CRITIC functional methods."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.functional.critic import (
    _prompt_agent,
    _prompt_critique,
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


def test__prompt_critique() -> None:
    """Test _prompt_critique function."""
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]), question="", examples="", answer=""
    )
    assert out == "1"

    # Test custom prompt.
    out = _prompt_critique(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        examples="",
        answer="",
        prompt="{question}{examples}",
    )
    assert out == "1"
