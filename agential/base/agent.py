"""Base agent interface class."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base agent class providing a general interface for agent operations."""

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally resets the agent's state."""
        raise NotImplementedError("Reset method not implemented.")

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally implementable method to generate a response."""
        raise NotImplementedError("Generate method not implemented.")
