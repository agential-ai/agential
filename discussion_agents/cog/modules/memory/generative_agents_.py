"""Generative Agents memory module implementation adapted from LangChain.

Original Paper: https://arxiv.org/abs/2304.03442
LangChain: https://github.com/langchain-ai/langchain
LangChain Generative Agents:
https://github.com/langchain-ai/langchain/tree/master/libs/experimental/langchain_experimental/generative_agents
LangChain Generative Agents Doc Page:
https://python.langchain.com/docs/use_cases/more/agents/agent_simulations/characters
"""
from datetime import datetime
from typing import List, Optional, Union

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents.base import Document

from discussion_agents.cog.modules.memory.base import BaseMemory


class GenerativeAgentMemory(BaseMemory):
    retriever: TimeWeightedVectorStoreRetriever

    def clear(self, retriever: TimeWeightedVectorStoreRetriever):
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

    def delete_memories(self):
        pass

    def load_memories(self):
        pass
