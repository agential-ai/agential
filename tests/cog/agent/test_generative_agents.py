"""Unit tests for GenerativeAgent."""
from langchain.llms.fake import FakeListLLM
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from discussion_agents.cog.agent.generative_agents import GenerativeAgent
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory

def test_init(time_weighted_retriever: TimeWeightedVectorStoreRetriever) -> None:
    """Test GenerativeAgent initialization."""
    memory = GenerativeAgentMemory(time_weighted_retriever)
    agent = GenerativeAgent(llm=FakeListLLM(responses=["1"]), memory=memory)
    assert agent