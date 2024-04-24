"""Base reflecting module."""

from abc import ABC, abstractmethod
from typing import Any


class BaseReflector(ABC):
    """Base reflecting class."""

    def __init__(self, llm: Any) -> None:
        """Initialization."""
        self.llm = llm

    @abstractmethod
    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        """Reflect on memory_contents w.r.t. relevant memories and returns a list of reflections."""
        pass

    @abstractmethod
    def clear(self, *args: Any, **kwargs: Any) -> Any:
        """Clears any internal state."""
        pass
