"""Utility methods for fetching memories."""
from datetime import datetime
from typing import List, Optional

from langchain.schema import BaseRetriever, Document
from langchain.utils import mock_now


def fetch_memories(
    memory_retriever: BaseRetriever,
    observation: str,
    now: Optional[datetime] = None,
) -> List[Document]:
    """Retrieve relevant memories based on the provided observation.

    Args:
        memory_retriever (TimeWeightedVectorStoreRetriever):
            The retriever used to access memory data.
        observation (str):
            The observation or query used to fetch related memories.
        now (Optional[datetime], optional):
            The current date and time for temporal context (default: None).

    Returns:
        List[Document]:
            A list of relevant documents representing memories.
    """
    if now is not None:
        with mock_now(now):
            return memory_retriever.get_relevant_documents(observation)
    else:
        return memory_retriever.get_relevant_documents(observation)
