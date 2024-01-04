"""Base scoring module."""
from abc import ABC, abstractmethod
from typing import Any, List, Union

from pydantic.v1 import BaseModel


class BaseScorer(BaseModel, ABC):
    """Base scoring class."""

    llm: Any

    @abstractmethod
    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[float]:
        """Score memory_contents with respect to relevant memories and returns a list of scores."""
        pass
