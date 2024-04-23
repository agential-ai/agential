"""Unit tests for utility fetch memory functions."""
from datetime import datetime

from langchain.retrievers import TimeWeightedVectorStoreRetriever

from agential.utils.fetch import fetch_memories

test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)


def test_fetch_memories(
    time_weighted_retriever: TimeWeightedVectorStoreRetriever,
) -> None:
    """Test fetch_memories."""
    observation = "Some observation."

    memories = fetch_memories(
        observation=observation,
        memory_retriever=time_weighted_retriever,
        now=test_date,
    )
    assert type(memories) is list
