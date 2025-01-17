"""Base strategy class for agents and prompting methods."""

from abc import ABC, abstractmethod
from typing import Any

from agential.core.llm import BaseLLM


class BaseStrategy(ABC):
    """An abstract base class for defining agent/prompting method strategies for generating responses.

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
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Resets the strategy's internal state, if any.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Reset optional values stored in input variables.
        """
        raise NotImplementedError
