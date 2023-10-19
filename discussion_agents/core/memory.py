"""Memory-related cores."""
from discussion_agents.core.base import BaseCore

from langchain.schema.retriever import BaseRetriever
from langchain.schema.memory import BaseMemory
from langchain.chains import LLMChain

class BaseCoreWithMemory(BaseCore):
    retriever: BaseRetriever
    memory: BaseMemory

    def chain(self, prompt: str) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            llm_kwargs=self.llm_kwargs,
            prompt=prompt,
            memory=self.memory
        )