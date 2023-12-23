"""Base scoring module."""
from abc import ABC, abstractmethod
from typing import List, Union

from pydantic import BaseModel

from langchain_core.language_models import LLM


class BaseScoring(BaseModel, ABC):
    """Base scoring class."""

    llm: LLM

    @abstractmethod
    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
    ) -> List[float]:
        """Score memory_contents with respect to relevant memories and returns a list of scores."""
        pass
