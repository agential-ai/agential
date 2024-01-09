"""Base reflecting module."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseReflector(BaseModel, ABC):
    """Base reflecting class."""

    llm: Any

    @abstractmethod
    def reflect(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Reflect on memory_contents w.r.t. relevant memories and returns a list of reflections."""
        pass
