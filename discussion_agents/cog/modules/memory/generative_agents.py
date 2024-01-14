"""Generative Agents memory module implementation adapted from LangChain.

This implementation only includes the necessary I/O for the retriever state.

Original Paper: https://arxiv.org/abs/2304.03442
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents.base import Document
from langchain_core.language_models import LLM

from discussion_agents.cog.modules.memory.base import BaseMemory
from discussion_agents.utils.fetch import fetch_memories


class GenerativeAgentMemory(BaseMemory):
    """A memory management class for Generative Agents, interfacing with a time-weighted vector store retriever.

    This class extends BaseMemory and manages the storage and retrieval of
    memories using a TimeWeightedVectorStoreRetriever. It provides
    functionalities to clear, add, and load memories in the context of
    generative agents' operations.

    Attributes:
        retriever (TimeWeightedVectorStoreRetriever): A retriever for handling memory storage and retrieval operations.
    """
    def __init__(self, retriever: TimeWeightedVectorStoreRetriever) -> None:
        """Initialization."""
        super().__init__()
        self.retriever = retriever

    def clear(self, retriever: TimeWeightedVectorStoreRetriever) -> None:
        """Clears the current retriever and sets it to a new one.

        Args:
            retriever (TimeWeightedVectorStoreRetriever): The new retriever to be set for handling memory operations.
        """
        self.retriever = retriever

    def add_memories(
        self,
        memory_contents: Union[str, List[str]],
        importance_scores: Union[float, List[float]],
        now: Optional[datetime] = None,
    ) -> None:
        """Adds a list of memories to the retriever with corresponding importance scores.

        Each memory content is associated with an importance score. The method validates the matching lengths of memory contents and importance scores and then adds these memories to the retriever.

        Args:
            memory_contents (Union[str, List[str]]): The memory contents to be added. memory_contents and importance scores
                must of the same length.
            importance_scores (Union[float, List[float]]): The importance scores corresponding to each memory content.
            now (Optional[datetime], optional): The current time, used for time-weighting the memories. Defaults to None.

        Raises:
            ValueError: If the lengths of memory_contents and importance_scores do not match.
        """
        if isinstance(memory_contents, str):
            memory_contents = [memory_contents]

        if isinstance(importance_scores, float):
            importance_scores = [importance_scores]

        if len(memory_contents) != len(importance_scores):
            raise ValueError(
                "The length of memory_contents must match the length of importance_scores."
            )

        documents = []
        for i in range(len(memory_contents)):
            documents.append(
                Document(
                    page_content=memory_contents[i],
                    metadata={"importance": importance_scores[i]},
                )
            )

        _ = self.retriever.add_documents(documents, current_time=now)

    def load_memories(
        self,
        queries: Optional[Union[str, List[str]]] = None,
        last_k: Optional[int] = None,
        consumed_tokens: Optional[int] = None,
        max_tokens_limit: Optional[int] = None,
        llm: Optional[LLM] = None,
        now: Optional[datetime] = None,
        queries_key: str = "relevant_memories",
        most_recent_key: str = "most_recent_memories",
        consumed_tokens_key: str = "most_recent_memories_limit",
    ) -> Dict[str, Any]:
        """Loads memories based on various criteria such as queries, recency, and token consumption limits.

        This method retrieves memories based on 3 options: provided queries, the most recent memories, or a token consumption limit.
        It can combine these criteria for a comprehensive retrieval.

        Args:
            queries (Optional[Union[str, List[str]]], optional): Queries to filter memories. Defaults to None.
            last_k (Optional[int], optional): The number of most recent memories to retrieve. Defaults to None.
            consumed_tokens (Optional[int], optional): The token limit for fetching memories.
                If consumed_tokens is not `None`, then max_tokens_limit and llm must be defined. Defaults to None.
            max_tokens_limit (Optional[int], optional): The maximum token limit for memory retrieval. Defaults to None.
            llm (Optional[LLM], optional): The language model used for token calculations. Defaults to None.
            now (Optional[datetime], optional): The current time, used for time-weighting. Defaults to None.
            queries_key (str, optional): Key for storing queried memories. Defaults to "relevant_memories".
            most_recent_key (str, optional): Key for storing most recent memories. Defaults to "most_recent_memories".
            consumed_tokens_key (str, optional): Key for storing memories within token limit. Defaults to "most_recent_memories_limit".

        Raises:
            ValueError: If consumed_tokens is defined but max_tokens_limit or llm is not.

        Returns:
            Dict[str, Any]: A dictionary containing retrieved memories categorized by the specified criteria.

        Example Output:
        The output dictionary can be any combination of the 3 keys: queries_key, most_recent_key,
        and consumed_tokens_key.

        {
            "relevant_memories": [Document, ..., Document],
            "most_recent_memories": [Document, ..., Document],
            "most_recent_memories_limit": [Document, ..., Document]
        }
        """
        if isinstance(queries, str):
            queries = [queries]

        if consumed_tokens and (not max_tokens_limit or not llm):
            raise ValueError(
                "max_tokens_limit and llm must be defined if consumed_tokens is defined."
            )

        memories = {}
        if queries:
            relevant_memories = [
                mem
                for query in queries
                for mem in fetch_memories(
                    observation=query,
                    memory_retriever=self.retriever,
                    now=now,
                )
            ]
            memories.update({queries_key: relevant_memories})

        if last_k:
            most_recent_memories = self.retriever.memory_stream[-last_k:]
            memories.update({most_recent_key: most_recent_memories})

        if consumed_tokens:
            results = []
            for doc in self.retriever.memory_stream[::-1]:  # type: ignore
                if max_tokens_limit and consumed_tokens >= max_tokens_limit:
                    break
                if llm:
                    consumed_tokens += llm.get_num_tokens(doc.page_content)
                if max_tokens_limit and consumed_tokens < max_tokens_limit:
                    results.append(doc)
            memories.update({consumed_tokens_key: results})

        return memories

    def show_memories(self, memories_key: str = "memory_stream") -> Dict[str, Any]:
        """Retrieves all stored memories and returns them in a dictionary.

        Args:
            memories_key (str, optional): The key under which the memories are stored. Defaults to "memory_stream".

        Returns:
            Dict[str, Any]: A dictionary containing the stored memories, keyed by `memories_key`.
        """
        return {memories_key: self.retriever.memory_stream}
