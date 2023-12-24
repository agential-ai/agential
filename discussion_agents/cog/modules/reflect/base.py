"""Base reflecting module."""
from abc import ABC, abstractmethod
from typing import List, Union

from langchain_core.language_models import LLM
from pydantic.v1 import BaseModel


class BaseReflector(BaseModel, ABC):
    """Base reflecting class."""

    llm: LLM

    @abstractmethod
    def reflect(
        self,
        observations: Union[str, List[str]],
    ) -> List[List[str]]:
        """Reflect on memory_contents w.r.t. relevant memories and returns a list of reflections."""
        pass
