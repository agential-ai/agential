"""Generic base Selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSelector(ABC):
    """An abstract base class for auto-selecting prompts and few-shot examples."""

    def __init__(self) -> None:
        """Initialization."""

    @abstractmethod
    def get_fewshots(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Gets the few-shot examples."""
        pass

    @abstractmethod
    def get_prompt(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        """Gets the prompt instruction."""
        pass
