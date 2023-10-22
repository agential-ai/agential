"""Base memory interface class."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union


class BaseMemoryInterface(ABC):
    """Requires memory classes to have an add_memories method."""

    @abstractmethod
    def add_memories(
        self, memory_contents: Union[str, List[str]], now: Optional[datetime] = None
    ) -> List[str]:
        """Memory classes must have some way to add memories to its memory bank."""
        pass
