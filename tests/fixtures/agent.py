"""Fixtures for creating agents."""

import pytest
import yaml

from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.agent.generative_agents import GenerativeAgent
from discussion_agents.cog.agent.react import ReActAgent
from discussion_agents.cog.agent.reflexion import ReflexionCoTAgent, ReflexionReActAgent


@pytest.fixture
def generative_agent() -> GenerativeAgent:
    """Creates a GenerativeAgent."""
    agent = GenerativeAgent(llm=FakeListLLM(responses=["1"]))
    return agent


@pytest.fixture
def react_agent() -> ReActAgent:
    """Creates a ReActAgent."""
    agent = ReActAgent(llm=FakeListLLM(responses=["1"]))
    return agent


@pytest.fixture
def reflexion_cot_agent() -> ReflexionCoTAgent:
    """Creates a ReflexionCoTAgent."""
    agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    return agent


@pytest.fixture
def reflexion_react_agent() -> ReflexionReActAgent:
    """Creates a ReflexionReActAgent."""
    agent = ReflexionReActAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    return agent


@pytest.fixture
def alfworld_env():
    """Prepare for env init for Alfworld."""
    with open(alfworld_config()) as reader:
        config = yaml.safe_load(reader) 
    return config

@pytest.fixture
def alfworld_config():
    return 'tests/assets/base_config.yaml'