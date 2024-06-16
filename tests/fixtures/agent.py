"""Fixtures for creating agents."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.agent.expel import ExpeLAgent


@pytest.fixture
def expel_agent() -> ExpeLAgent:
    """Creates a ExpeLAgent."""
    agent = ExpeLAgent(llm=FakeListChatModel(responses=["1"]))
    return agent
