"""Memory-related cores."""
from langchain.chains import LLMChain
from langchain.schema.memory import BaseMemory
from langchain.schema.prompt_template import BasePromptTemplate

from discussion_agents.core.base import BaseCore


class BaseCoreWithMemory(BaseCore):
    """Agent core class with memory support and a retriever, extending the BaseCore.

    This class extends BaseCore, enforcing a retriever and memory component
    as arguments. Additionally, it uses the memory in the LLMChain.

    Attributes:
        llm (BaseLanguageModel): The language model used by the agent's core.
        llm_kwargs (Dict[str, Any], optional): Additional keyword arguments for configuring the language model.
        retriever (BaseRetriever, optional): A retriever component for accessing external data.
        memory (BaseMemory): A memory management component for storing and retrieving agent memories.

    Methods:
        chain(prompt: str) -> LLMChain:
            Create an LLMChain instance for generating responses based on the provided prompt.
    """

    memory: BaseMemory

    def chain(self, prompt: BasePromptTemplate) -> LLMChain:
        """Create an LLMChain based on a given prompt template.

        BaseCoreWithMemoryreturns a stateful LLMChain (with memory).

        Args:
            prompt (BasePromptTemplate): The prompt template used to initialize the LLMChain.

        Returns:
            LLMChain: An instance of the LLMChain with the specified prompt template.
        """
        return LLMChain(
            llm=self.llm, llm_kwargs=self.llm_kwargs, prompt=prompt, memory=self.memory
        )
