"""Memory-related cores."""
from langchain.chains import LLMChain
from langchain.schema.memory import BaseMemory
from langchain.schema.retriever import BaseRetriever

from discussion_agents.core.base import BaseCore


class BaseCoreWithMemory(BaseCore):
    """
    Agent core class with memory support and a retriever, extending the BaseCore.

    This class extends BaseCore, enforcing a retriever and memory component
    as arguments. Additionally, it uses the memory in the LLMChain.

    Attributes:
        llm (BaseLanguageModel): The language model used by the agent's core.
        llm_kwargs (Dict[str, Any], optional): Additional keyword arguments for configuring the language model.
        retriever (BaseRetriever): A retriever component for accessing external data.
        memory (BaseMemory): A memory management component for storing and retrieving agent memories.

    Methods:
        chain(prompt: str) -> LLMChain:
            Create an LLMChain instance for generating responses based on the provided prompt.
    """

    retriever: BaseRetriever
    memory: BaseMemory

    def chain(self, prompt: str) -> LLMChain:
        return LLMChain(
            llm=self.llm, llm_kwargs=self.llm_kwargs, prompt=prompt, memory=self.memory
        )
