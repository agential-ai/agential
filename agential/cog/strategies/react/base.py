"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.strategies.base import BaseStrategy


class ReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReAct Agent."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        pass

    @abstractmethod
    def generate_observation(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        pass


    @abstractmethod
    def create_output_dict(self, answer: str, critique: str) -> Dict[str, str]:
        pass

    @abstractmethod
    def halting_condition(self) -> bool:
        pass
