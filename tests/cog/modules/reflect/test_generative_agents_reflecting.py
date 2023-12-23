"""Unit tests for Generative Agents reflecting module."""
from datetime import datetime

from langchain.llms.fake import FakeListLLM
from langchain_core.retrievers import BaseRetriever

from discussion_agents.cog.modules.reflect.generative_agents import (
    GenerativeAgentReflector,
)

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)


def test_generative_agent_scorer(memory_retriever: BaseRetriever) -> None:
    """Test GenerativeAgentReflector."""
    observations = "Chairs have 4 legs."
    llm = FakeListLLM(responses=["That's an interesting observation!"])
    reflector = GenerativeAgentReflector(
        llm=llm, retriever=memory_retriever, importance_weight=0.15
    )
    insights = reflector.reflect(observations=observations, now=test_date)
    assert type(insights) is list
    assert insights == [["That's an interesting observation!"]]
