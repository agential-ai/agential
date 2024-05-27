"""Unit tests for Self-Refine."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.self_refine import SelfRefineAgent
from agential.cog.strategies.self_refine.base import SelfRefineBaseStrategy


def test_init() -> None:
    """Test initialization."""
    agent = SelfRefineAgent(
        llm=FakeListChatModel(responses=[]), 
        mode={"math": "gsm8k"}
    )
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.strategy, SelfRefineBaseStrategy)
    assert agent.mode == {"math": "gsm8k"}


def test_reset() -> None:
    """Test reset."""
    agent = SelfRefineAgent(
        llm=FakeListChatModel(responses=[]), 
        mode={"math": "gsm8k"}
    )
    agent.strategy._halt = True
    agent.reset()
    assert not agent.strategy._halt


def test_generate() -> None:
    """Test generate."""
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"

