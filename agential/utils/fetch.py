"""Utility methods for fetching memories."""
from datetime import datetime
from typing import List, Optional

from langchain.schema import BaseRetriever, Document
from langchain.utils import mock_now


def fetch_memories(
    observation: str,
    memory_retriever: BaseRetriever,
    now: Optional[datetime] = None,
) -> List[Document]:
    """Retrieve relevant memories based on the observation and time.

    Args:
        observation (str):
            The observation or query used to fetch related memories.
        memory_retriever (BaseRetriever):
            The retriever used to access memory data.
        now (Optional[datetime], optional):
            The current date and time for temporal context (default: None).

    Returns:
        List[Document]:
            A list of relevant documents representing memories.

    Example:
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, otherScoreKeys=["importance"], k=5
        )
        result = fetch_memories("An observation", retriever, now=datetime.now())
    """
    if now is not None:
        with mock_now(now):
            return memory_retriever.get_relevant_documents(observation)
    else:
        return memory_retriever.get_relevant_documents(observation)
