"""Generic base strategy class."""

from abc import ABC, abstractmethod
from typing import Any

from agential.llm.llm import BaseLLM


class BaseAgentStrategy(ABC):
    """An abstract base class for defining strategies for generating responses with LLM-based agents.

    Parameters:
        llm (BaseLLM): An instance of a language model used for generating responses.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        self.llm = llm
        self.testing = testing

    @abstractmethod
    def generate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Generates a response.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The generated response.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Resets the strategy's internal state, if any.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError
