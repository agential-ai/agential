"""Fixtures for creating agents."""

import pytest

from langchain.llms.fake import FakeListLLM
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from discussion_agents.cog.agent.generative_agents import GenerativeAgent
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory

@pytest.fixture
def generative_agents(time_weighted_retriever: TimeWeightedVectorStoreRetriever) -> GenerativeAgent:
    """Creates a GenerativeAgent."""
    memory = GenerativeAgentMemory(retriever=time_weighted_retriever)
    agent = GenerativeAgent(llm=FakeListLLM(responses=["1"]), memory=memory)
    return agent