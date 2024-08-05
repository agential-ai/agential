"""Base LLM class."""

from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Base LLM class."""

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """Generate text based on a prompt."""
        pass

    @abstractmethod
    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens used in a text."""
        pass