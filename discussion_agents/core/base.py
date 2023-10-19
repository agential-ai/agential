"""Base agent core."""
from typing import Dict, Any

from pydantic.v1 import BaseModel, Field
from abc import ABC, abstractmethod

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.memory import BaseMemory
from langchain.schema.retriever import BaseRetriever
from langchain.chains import LLMChain

class BaseCoreInterface(BaseModel, ABC):
    @abstractmethod
    def chain(self, prompt: str) -> LLMChain:
        pass

class BaseCore(BaseCoreInterface):
    llm: BaseLanguageModel
    llm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    retriever: BaseRetriever = Field(default=None)
    memory: BaseMemory = Field(default=None)

    def chain(self, prompt: str) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            llm_kwargs=self.llm_kwargs,
            prompt=prompt
        )