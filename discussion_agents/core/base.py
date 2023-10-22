"""Base agent core."""
from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.memory import BaseMemory
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.schema.retriever import BaseRetriever
from pydantic.v1 import BaseModel, Field


class BaseCoreInterface(BaseModel, ABC):
    """Requires BaseCore and subclasses to have a chain method."""

    @abstractmethod
    def chain(self, prompt: BasePromptTemplate) -> LLMChain:
        """Abstract method for creating an LLMChain based on a given prompt.

        Args:
            prompt (BasePromptTemplate): The prompt used to initialize the LLMChain.

        Returns:
            LLMChain: An instance of the LLMChain with the specified prompt.
        """
        pass


class BaseCore(BaseCoreInterface):
    """Base class for an agent core that interfaces with language models, retrievers, and memory.

    This base class defines the structure of an agent's core component, which serves as a bridge
    between various components such as language models, retrievers, and memory management. It
    provides methods for creating and managing LLMChain instances for generating responses
    and insights.

    Note:
        Subclasses of this base core class should provide specific implementations for
        their respective use cases, customizing the interaction between language models,
        retrievers, and memory management.

    Attributes:
        llm (BaseLanguageModel): The language model used by the agent's core.
        llm_kwargs (Dict[str, Any], optional): Additional keyword arguments for configuring the language model.
        retriever (BaseRetriever, optional): A retriever component for accessing external data.
        memory (BaseMemory, optional): A memory management component for storing and retrieving agent memories.
    """

    llm: BaseLanguageModel
    llm_kwargs: Dict[str, Any] = Field(default_factory=dict)
    retriever: BaseRetriever = Field(default=None)
    memory: BaseMemory = Field(default=None)

    def chain(self, prompt: BasePromptTemplate) -> LLMChain:
        """Create an LLMChain based on a given prompt template.

        BaseCore only returns stateless LLMChains.

        Args:
            prompt (BasePromptTemplate): The prompt template used to initialize the LLMChain.

        Returns:
            LLMChain: An instance of the LLMChain with the specified prompt template.
        """
        return LLMChain(llm=self.llm, llm_kwargs=self.llm_kwargs, prompt=prompt)
