"""Generative Agents memory module implementation adapted from LangChain.

Original Paper: https://arxiv.org/abs/2304.03442
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

from langchain_core.language_models import LLM
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents.base import Document

from discussion_agents.utils.fetch import fetch_memories
from discussion_agents.cog.modules.memory.base import BaseMemory


class GenerativeAgentMemory(BaseMemory):
    retriever: TimeWeightedVectorStoreRetriever
    queries_key: str = "relevant_memories"
    most_recent_key: str = "most_recent_memories"
    consumed_tokens_key: str = "most_recent_memories_limit"

    def clear(self, retriever: TimeWeightedVectorStoreRetriever) -> None:
        self.retriever = retriever

    def add_memories(
        self,
        memory_contents: Union[str, List[str]],
        importance_scores: Union[float, List[float]],
        now: Optional[datetime] = None,
    ) -> None:
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
        llm: LLM = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        if isinstance(queries, str):
            queries = [queries]
        
        if consumed_tokens and (not max_tokens_limit and not llm): 
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
            memories.update({self.queries_key: relevant_memories})

        if last_k:
            most_recent_memories = self.retriever.memory_stream[-last_k:]
            memories.update({self.most_recent_key: most_recent_memories})

        if consumed_tokens:
            results = []
            for doc in self.retriever.memory_stream[::-1]:  # type: ignore
                if consumed_tokens >= max_tokens_limit:
                    break
                consumed_tokens += llm.get_num_tokens(doc.page_content)
                if consumed_tokens < max_tokens_limit:
                    results.append(doc)
            memories.update({self.consumed_tokens_key: results})

        return memories