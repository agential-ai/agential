"""Base memory interface class."""
from datetime import datetime
from typing import List, Union, Optional
from abc import ABC, abstractmethod

class AddMemoriesInterface(ABC):
    @abstractmethod
    def add_memories(self, memory_contents: Union[str, List[str]], now: Optional[datetime] = None) -> List[str]:
        pass