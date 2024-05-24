"""Generic base strategy class."""

from abc import ABC, abstractmethod
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel


class BaseStrategy(ABC):
    """An abstract base class for defining strategies for generating responses with LLM-based agents."""

    @abstractmethod
    def generate(
        self,
        llm: BaseChatModel,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Generates a response using the provided language model, question, examples, and prompt.

        Args:
            llm (BaseChatModel): The language model to be used for generating the response.
            question (str): The question to be answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the response.
            prompt (str): The instruction template used to prompt the language model.
            additional_keys (Dict[str, str]): Additional keys to format the prompt.

        Returns:
            str: The generated response.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the strategy's internal state, if any.

        This method should be implemented to clear any internal state that the strategy maintains
        between generations, preparing it for a new sequence of interactions.

        Returns:
            None
        """
        pass
