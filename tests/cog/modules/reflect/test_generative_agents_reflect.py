"""Unit tests for Generative Agents reflect module."""
from datetime import datetime

from langchain.llms.fake import FakeListLLM
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from discussion_agents.cog.modules.reflect.generative_agents import (
    GenerativeAgentReflector,
)

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)


def test_generative_agent_reflector(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    """Test GenerativeAgentReflector."""
    observations = "Chairs have 4 legs."
    llm = FakeListLLM(responses=["That's an interesting observation!"])
    reflector = GenerativeAgentReflector(llm=llm, retriever=time_weighted_retriever)
    insights = reflector.reflect(observations=observations, now=test_date)
    assert type(insights) is list
    assert insights == [["That's an interesting observation!"]]


def test_generative_agent_reflector_clear(time_weighted_retriever: TimeWeightedVectorStoreRetriever) -> None:
    """Test GenerativeAgentReflector clear method."""
    llm = FakeListLLM(responses=["That's an interesting observation!"])
    reflector = GenerativeAgentReflector(llm=llm, retriever=time_weighted_retriever)
    reflector.clear(time_weighted_retriever)
    assert reflector.retriever