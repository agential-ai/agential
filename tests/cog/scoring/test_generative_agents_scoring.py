"""Testing Generative Agents memory scoring method(s)."""

from langchain.llms.fake import FakeListLLM
from discussion_agents.cog.functional.generative_agents import score_memories_importance

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