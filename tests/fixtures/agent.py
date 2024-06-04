"""Fixtures for creating agents."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.agent.expel import ExpeLAgent
from agential.cog.agent.reflexion import ReflexionReActAgent


@pytest.fixture
def reflexion_react_agent() -> ReflexionReActAgent:
    """Creates a ReflexionReActAgent."""
    agent = ReflexionReActAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
        max_trials=1,
        max_reflections=3,
    )
    return agent


@pytest.fixture
def expel_agent() -> ExpeLAgent:
    """Creates a ExpeLAgent."""
    agent = ExpeLAgent(llm=FakeListChatModel(responses=["1"]))
    return agent
