"""Generic base Selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from agential.base.strategies import BaseStrategy


class BaseFactory(ABC):
    """Base factory class for creating strategy instances and auto-selecting prompts and few-shot examples."""

    def __init__(self) -> None:
        """Initialize the BaseFactory class."""
        pass

    @abstractmethod
    def get_fewshots(self, benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve few-shot examples based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of few-shot examples.
        """
        pass

    @abstractmethod
    def get_prompt(self, benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        pass

    @abstractmethod
    def get_strategy(self, benchmark: str, **strategy_kwargs: Any) -> BaseStrategy:
        """Returns an instance of the appropriate strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            BaseStrategy: An instance of the appropriate strategy.
        """
        pass
