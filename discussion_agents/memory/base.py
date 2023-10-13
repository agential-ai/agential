"""Base memory interface class."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union

from langchain.schema import BaseRetriever, Document


class BaseMemoryInterface(ABC):
    @abstractmethod
    def add_memories(
        self, memory_contents: Union[str, List[str]], now: Optional[datetime] = None
    ) -> List[str]:
        pass

    @abstractmethod
    def fetch_memories(
        self,
        observation: str,
        now: Optional[datetime] = None,
    ) -> List[Document]:
        pass
