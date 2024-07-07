"""Generic base Selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSelector(ABC):
    """An abstract base class for auto-selecting prompts and few-shot examples."""

    def __init__(self) -> None:
        """Initialize the BaseSelector class."""
        pass

    @abstractmethod
    def get_fewshots(
        self,
        benchmark: str,
        **kwargs: Any
    ) -> Dict[str, str]:
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
