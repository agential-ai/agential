"""Generic base strategy class."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from agential.llm.llm import BaseLLM


class BaseStrategy(ABC):
    """An abstract base class for defining strategies for generating responses with LLM-based agents."""

    def __init__(self, llm: BaseLLM) -> None:
        """Initialization."""
        self.llm = llm

    @abstractmethod
    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Generates a response."""
        pass

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Resets the strategy's internal state, if any."""
        pass