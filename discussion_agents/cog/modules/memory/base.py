"""Base memory interface class."""
from abc import ABC, abstractmethod

from pydantic.v1 import BaseModel


class BaseMemory(BaseModel, ABC):
    """Base memory class providing a general interface for memory operations."""

    @abstractmethod
    def clear(self):
        """Clear all memories.

        Implementations should override this method to provide the functionality
        to clear memories. Specific parameters and return types depend on the implementation.
        """
        pass

    @abstractmethod
    def add_memories(self):
        """Add memories.

        Implementations should override this method to provide the functionality
        to add memories. Specific parameters and return types depend on the implementation.
        """
        pass

    @abstractmethod
    def load_memories(self):
        """Load memories.

        Implementations should override this method to provide the functionality
        to load memories. Specific parameters and return types depend on the implementation.
        """
        pass