"""Base scoring module."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from langchain_core.language_models import LLM


class BaseScoring(ABC):
    """Base scoring class."""

    llm: LLM
    llm_kwargs: Dict[str, Any] = {}

    @abstractmethod
    def score(
        self,
        memory_contents: Union[str, List[str]],
        relevant_memories: Union[str, List[str]],
    ) -> List[str]:
        """Score memory_contents with respect to relevant memories."""
        pass
