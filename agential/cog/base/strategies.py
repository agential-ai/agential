"""Generic base strategy class."""

from abc import ABC, abstractmethod
from typing import Any

from agential.llm.llm import BaseLLM


class BaseStrategy(ABC):
    """An abstract base class for defining strategies for generating responses with LLM-based agents."""

    def __init__(self, llm: BaseLLM) -> None:
        """Initialization."""
        self.llm = llm

    @abstractmethod
    def generate(
        self,
        testing: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Generates a response.
        
        Args:
            testing (bool): Whether the strategy is being used for testing.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The generated response.
        """
        pass

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Resets the strategy's internal state, if any.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass
