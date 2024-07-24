"""Base LATS Agent strategy class."""

from abc import abstractmethod

from agential.base.strategies import BaseStrategy
from langchain_core.language_models.chat_models import BaseChatModel


class LATSBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the LATS Agent."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

    @abstractmethod
    def select_node():
        pass

    @abstractmethod
    def expand_node():
        pass

    @abstractmethod
    def evaluate_node():
        pass

    @abstractmethod
    def simulate_note():
        pass

    @abstractmethod
    def backpropagate_node():
        pass

    @abstractmethod
    def reflect_node():
        pass