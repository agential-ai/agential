"""Unit tests for Generative Agents scoring module."""
from langchain.llms.fake import FakeListLLM

from discussion_agents.cog.modules.scoring.generative_agents import (
    GenerativeAgentScorer,
)


def test_generative_agent_scorer():
    """Test GenerativeAgentScorer."""
    llm = FakeListLLM(responses=["1"])
    scorer = GenerativeAgentScorer(llm=llm, importance_weight=0.15)
    scores = scorer.score(
        memory_contents="Some memory.", relevant_memories="some relevant memories."
    )
    assert type(scores) is list
    assert len(scores) == 1
    assert type(scores[0]) == float
    assert scores[0] == 0.015
