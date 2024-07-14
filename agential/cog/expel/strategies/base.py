"""Base ExpeL Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.strategies import BaseStrategy


class ExpeLBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ExpeL Agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

    @abstractmethod
    def get_dynamic_examples(
        self,
        question: str,
        examples: str,
        k_docs: int,
        num_fewshots: int,
        max_fewshot_tokens: int,
        reranker_strategy: str,
        additional_keys: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str]]:
        pass

    @abstractmethod
    def gather_experience(self):
        pass

    @abstractmethod
    def extract_insights(self):
        pass

    @abstractmethod
    def update_insights(self):
        pass
