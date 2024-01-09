"""Base memory interface class."""
from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class BaseMemory(BaseModel, ABC):
    """Base memory class providing a general interface for memory operations."""

    @abstractmethod
    def clear(self, *args: Any, **kwargs: Any) -> None:
        """Clear all memories.

        Implementations should override this method to provide the functionality
        to clear memories. Specific parameters and return types depend on the implementation.
        """
        pass

    @abstractmethod
    def add_memories(self, *args: Any, **kwargs: Any) -> None:
        """Add memories.

        Implementations should override this method to provide the functionality
        to add memories. Specific parameters and return types depend on the implementation.
        """
        pass

    @abstractmethod
    def load_memories(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Load memories and return a dictionary.

        Implementations should override this method to provide the functionality
        to load memories. Specific parameters and return types depend on the implementation.
        """
        pass

    @abstractmethod
    def show_memories(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Show all memories.

        Implementations should override this method to provide the functionality
        to show memories. Specific parameters and return types depend on the implementation.
        """
        pass
