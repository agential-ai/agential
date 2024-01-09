"""Base scoring module."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseScorer(BaseModel, ABC):
    """Base scoring class."""

    llm: Any

    @abstractmethod
    def score(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Score memory_contents with respect to relevant memories and returns a list of scores."""
        pass
