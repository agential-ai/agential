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


def test_get_topics_of_reflection(generative_agent: GenerativeAgent) -> None:
    """Test get_topics_of_reflection method."""
    out = generative_agent.get_topics_of_reflection(last_k=5)
    assert isinstance(out, list)
    assert out[0] == "1"


def test_get_insights_on_topic(generative_agent: GenerativeAgent) -> None:
    """Test get_insights_on_topic method."""
    out = generative_agent.get_insights_on_topic(topics=["Some topic."])
    assert isinstance(out, list)
    assert isinstance(out[0], list)
    assert out[0][0] == "1"


def test_reflect(generative_agent: GenerativeAgent) -> None:
    """Test reflect method."""
    insights = generative_agent.reflect(last_k=5)
    assert isinstance(insights, list)
    assert isinstance(insights[0], list)
    assert insights[0][0] == "1"


def test_add_memories(generative_agent: GenerativeAgent) -> None:
    """Test add_memories method."""
    generative_agent.add_memories(memory_contents="Some observation.", now=test_date)
    assert len(generative_agent.memory.retriever.memory_stream) == 1
    obs = generative_agent.memory.retriever.memory_stream
    assert obs[0].page_content == "Some observation."
    assert obs[0].metadata["importance"] == 0.015
    assert obs[0].metadata["last_accessed_at"] == test_date
    assert obs[0].metadata["created_at"] == test_date


def test_score(generative_agent: GenerativeAgent) -> None:
    """Test score method."""
    scores = generative_agent.score(
        memory_contents="Some topic.", relevant_memories=["Some other topics."]
    )
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert isinstance(scores[0], float)


def test_get_entity_from_observation(generative_agent: GenerativeAgent) -> None:
    """Test get_entity_from_observation method."""
    observation = "Observation about an entity."
    out = generative_agent.get_entity_from_observation(observation)
    assert out == "1"


def test_get_entity_action(generative_agent: GenerativeAgent) -> None:
    """Test get_entity_action method."""
    observation = "Observation about an entity doing something."
    entity_name = "EntityName"
    out = generative_agent.get_entity_action(observation, entity_name)
    assert out == "1"


def test_summarize_related_memories(generative_agent: GenerativeAgent) -> None:
    """Test summarize_related_memories method."""
    out = generative_agent.summarize_related_memories(observation="An observation.")
    assert out == "1"


def test_get_summary(generative_agent: GenerativeAgent) -> None:
    """Test get_summary method."""
    summary = generative_agent.get_summary(force_refresh=True)
    assert isinstance(summary, str)


def test__generate_reaction(generative_agent: GenerativeAgent) -> None:
    """Test _generate_reaction method."""
    out = generative_agent._generate_reaction(observation="An observation.", suffix="")
    assert isinstance(out, str)
    assert out == "1"


def test_generate_reaction(generative_agent: GenerativeAgent) -> None:
    """Test generate_reaction method."""
    out = generative_agent.generate_reaction(observation="An observation.")
    assert isinstance(out, tuple)
    assert not out[0]
    assert isinstance(out[1], str)


def test_generate_dialogue_response(generative_agent: GenerativeAgent) -> None:
    """Test generate_dialogue_response method."""
    out = generative_agent.generate_dialogue_response(observation="An observation.")
    assert isinstance(out, tuple)
    assert not out[0]
    assert isinstance(out[1], str)
