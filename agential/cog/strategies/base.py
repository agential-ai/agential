"""Generic base strategy class."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


class BaseStrategy(ABC):
    """An abstract base class for defining strategies for generating responses with LLM-based agents."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        self.llm = llm

    @abstractmethod
    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Generates a response."""
        pass

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Resets the strategy's internal state, if any."""
        pass
