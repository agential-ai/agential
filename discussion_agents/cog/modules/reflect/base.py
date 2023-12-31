"""Base reflecting module."""
from abc import ABC, abstractmethod
from typing import Any, List, Union

from pydantic.v1 import BaseModel


class BaseReflector(BaseModel, ABC):
    """Base reflecting class."""

    llm: Any

    @abstractmethod
    def reflect(
        self, observations: Union[str, List[str]], **kwargs: Any
    ) -> List[List[str]]:
        """Reflect on memory_contents w.r.t. relevant memories and returns a list of reflections."""
        pass
