from datetime import datetime
from typing import List, Optional

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain.utils import mock_now


def fetch_memories(
    memory_retriever: TimeWeightedVectorStoreRetriever,
    observation: str,
    now: Optional[datetime] = None,
) -> List[Document]:
    """Fetch related memories based on an observation."""
    if now is not None:
        with mock_now(now):
            return memory_retriever.get_relevant_documents(observation)
    else:
        return memory_retriever.get_relevant_documents(observation)
