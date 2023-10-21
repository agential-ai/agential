"""Memory-related cores."""
from langchain.chains import LLMChain
from langchain.schema.memory import BaseMemory
from langchain.schema.retriever import BaseRetriever

from discussion_agents.core.base import BaseCore


class BaseCoreWithMemory(BaseCore):
    retriever: BaseRetriever
    memory: BaseMemory

    def chain(self, prompt: str) -> LLMChain:
        return LLMChain(
            llm=self.llm, llm_kwargs=self.llm_kwargs, prompt=prompt, memory=self.memory
        )
