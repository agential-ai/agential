"""Base scoring module."""
from abc import ABC, abstractmethod
from typing import Any


class BaseScorer(ABC):
    """Base scoring class."""

    def __init__(self, llm: Any) -> None:
        """Initialization."""
        self.llm = llm

    @abstractmethod
    def score(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Score memory_contents with respect to relevant memories and returns a list of scores."""
        pass
