"""Unit tests for GenerativeAgent."""
from datetime import datetime
from langchain.llms.fake import FakeListLLM
from langchain.retrievers import TimeWeightedVectorStoreRetriever

from discussion_agents.cog.agent.generative_agents import GenerativeAgent
from discussion_agents.cog.modules.memory.generative_agents import GenerativeAgentMemory

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)

def test_init(time_weighted_retriever: TimeWeightedVectorStoreRetriever) -> None:
    """Test GenerativeAgent initialization."""
    memory = GenerativeAgentMemory(retriever=time_weighted_retriever)
    agent = GenerativeAgent(llm=FakeListLLM(responses=["1"]), memory=memory)
    assert agent
    assert agent.llm
    assert agent.memory
    assert agent.reflector
    assert agent.importance_weight == 0.15
    assert agent.reflection_threshold == 8

def test_get_topics_of_reflection(generative_agents: GenerativeAgent) -> None:
    """Test get_topics_of_reflection method."""
    out = generative_agents.get_topics_of_reflection(last_k=5)
    assert isinstance(out, list)
    assert out[0] == "1"

def test_get_insights_on_topic(generative_agents: GenerativeAgent) -> None:
    """Test get_insights_on_topic method."""
    out = generative_agents.get_insights_on_topic(topics=["Some topic."])
    assert isinstance(out, list)
    assert isinstance(out[0], list)
    assert out[0][0] == "1"
    
def test_reflect(generative_agents: GenerativeAgent) -> None:
    """Test reflect method."""
    insights = generative_agents.reflect(last_k=5)
    assert isinstance(insights, list)
    assert isinstance(insights[0], list)
    assert insights[0][0] == "1"

def test_add_memories(generative_agents: GenerativeAgent) -> None:
    """Test add_memories method."""
    generative_agents.add_memories(memory_contents="Some observation.", now=test_date)
    assert len(generative_agents.memory.retriever.memory_stream) == 1
    obs = generative_agents.memory.retriever.memory_stream
    assert obs[0].page_content == "Some observation."
    assert obs[0].metadata["importance"] == 0.015
    assert obs[0].metadata["last_accessed_at"] == test_date
    assert obs[0].metadata["created_at"] == test_date

def test_score(generative_agents: GenerativeAgent) -> None:
    scores = generative_agents.score(
        memory_contents="Some topic.", 
        relevant_memories=["Some other topics."]
    )
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert isinstance(scores[0], float)

def test_get_entity_from_observation(generative_agents: GenerativeAgent):
    observation = "Observation about an entity."
    out = generative_agents.get_entity_from_observation(observation)
    assert out == "1"

def test_get_entity_action(generative_agents: GenerativeAgent):
    observation = "Observation about an entity doing something."
    entity_name = "EntityName"
    out = generative_agents.get_entity_action(observation, entity_name)
    assert out == "1"
