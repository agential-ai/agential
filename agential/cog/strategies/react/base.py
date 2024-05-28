"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Dict, Tuple

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
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str]:
        pass

    @abstractmethod
    def generate_observation(
        self,
        action_type: str, 
        query: str
    ) -> str:
        pass


    @abstractmethod
    def create_output_dict(self, thought: str, action: str, obs: str) -> Dict[str, str]:
        pass

    @abstractmethod
    def halting_condition(self, action_type: str) -> bool:
        pass
