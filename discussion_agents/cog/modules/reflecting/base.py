"""Base reflecting module."""
from abc import ABC, abstractmethod
from typing import List, Union

from pydantic import BaseModel

from langchain_core.language_models import LLM


class BaseReflect(BaseModel, ABC):
    """Base reflecting class."""

    llm: LLM

    @abstractmethod
    def reflect(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
    ) -> List[str]:
        """Reflect on memory_contents w.r.t. relevant memories and returns a list of reflections."""
        pass

