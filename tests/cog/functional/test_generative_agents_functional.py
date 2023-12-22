"""Testing Generative Agents memory functional methods."""
from datetime import datetime

from langchain.llms.fake import FakeListLLM
from langchain.schema import BaseRetriever

from discussion_agents.cog.functional.generative_agents import (
    get_insights_on_topics,
    get_topics_of_reflection,
    reflect,
    score_memories_importance,
)

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)


def test_score_memories_importance() -> None:
    """Test score_memories_importance."""
    importance_weight = 0.15

    # Test memory_contents str relevant_memories str.
    scores = score_memories_importance(
        memory_contents="Some memory.",
        relevant_memories="some relevant memories.",
        llm=FakeListLLM(responses=["1"]),
        importance_weight=importance_weight,
    )
    assert type(scores) is list
    assert len(scores) == 1
    assert type(scores[0]) == float
    assert scores[0] == 0.015

    # Test memory_contents str relevant_memories list.
    scores = score_memories_importance(
        memory_contents="Some memory.",
        relevant_memories=["some relevant memories.", "some relevant memories."],
        llm=FakeListLLM(responses=["1"]),
        importance_weight=importance_weight,
    )
    assert type(scores) is list
    assert len(scores) == 1
    assert type(scores[0]) == float
    assert scores[0] == 0.015

    # Test memory_contents list relevant_memories str.
    scores = score_memories_importance(
        memory_contents=["Some memory.", "Some memory."],
        relevant_memories="some relevant memories.",
        llm=FakeListLLM(responses=["1"]),
        importance_weight=importance_weight,
    )
    assert type(scores) is list
    assert len(scores) == 2
    for i in scores:
        assert type(i) == float
        assert i == 0.015

    # Test memory_contents list relevant_memories list.
    scores = score_memories_importance(
        memory_contents=["Some memory."],
        relevant_memories=["some relevant memories."],
        llm=FakeListLLM(responses=["1"]),
        importance_weight=importance_weight,
    )
    assert type(scores) is list
    assert len(scores) == 1
    assert type(scores[0]) == float
    assert scores[0] == 0.015


def test_get_topics_of_reflection() -> None:
    """Tests get_topics_of_reflection."""
    llm = FakeListLLM(responses=["That's an interesting observation!"])

    # Test observations string.
    observations = "This is an observation."
    topics = get_topics_of_reflection(observations=observations, llm=llm)
    assert type(topics) is list
    assert topics == ["That's an interesting observation!"]

    # Test observations list.
    observations = ["This is an observation."]
    topics = get_topics_of_reflection(observations=observations, llm=llm)
    assert type(topics) is list
    assert topics == ["That's an interesting observation!"]


def test_get_insights_on_topics() -> None:
    """Tests get_insights_on_topics."""
    llm = FakeListLLM(responses=["That's an interesting observation!"])

    # Test topics list and related_memories list.
    insights = get_insights_on_topics(
        topics=["Some topic."],
        related_memories=["This is another topic."],
        llm=llm,
    )
    assert type(insights) is list
    assert type(insights[0]) is list
    assert len(insights) == 1
    assert len(insights[0]) == 1
    assert insights[0] == ["That's an interesting observation!"]

    # Test topics str and related_memories str.
    insights = get_insights_on_topics(
        topics="Some topic.",
        related_memories="This is another topic.",
        llm=llm,
    )
    assert type(insights) is list
    assert type(insights[0]) is list
    assert len(insights) == 1
    assert len(insights[0]) == 1
    assert insights[0] == ["That's an interesting observation!"]

    # Test topics str and related_memories list.
    insights = get_insights_on_topics(
        topics="Some topic.",
        related_memories=["This is another topic."],
        llm=llm,
    )
    assert type(insights) is list
    assert type(insights[0]) is list
    assert len(insights) == 1
    assert len(insights[0]) == 1
    assert insights[0] == ["That's an interesting observation!"]

    # Test topics list and related_memories str.
    insights = get_insights_on_topics(
        topics=["Some topic."],
        related_memories="This is another topic.",
        llm=llm,
    )
    assert type(insights) is list
    assert type(insights[0]) is list
    assert len(insights) == 1
    assert len(insights[0]) == 1
    assert insights[0] == ["That's an interesting observation!"]


def test_reflect(memory_retriever: BaseRetriever) -> None:
    """Tests reflect."""
    llm = FakeListLLM(responses=["That's an interesting observation!"])

    observations = "Chairs have 4 legs."
    topics, insights = reflect(
        observations=observations, llm=llm, retriever=memory_retriever, now=test_date
    )

    assert type(topics) is list
    assert topics == ["That's an interesting observation!"]
    assert type(insights) is list
    assert insights == [["That's an interesting observation!"]]
