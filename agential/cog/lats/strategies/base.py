"""Base LATS Agent strategy class."""

from typing import List, Dict
from abc import abstractmethod

from agential.base.strategies import BaseStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.lats.node import Node

class LATSBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the LATS Agent."""

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

    @abstractmethod
    def initialize(self) -> Node:
        pass

    @abstractmethod
    def generate(self, node: Node, question: str, key: str, examples: str, reflect_examples: str, prompt: str, reflect_prompt: str, additional_keys: Dict[str, str], reflect_additional_keys: Dict[str, str]) -> tuple:
        pass

    @abstractmethod
    def generate_thought(self, question: str, examples: str, trajectory: str, reflections: str, depth: int, prompt: str, additional_keys: Dict[str, str]) -> tuple:
        pass

    @abstractmethod
    def generate_action(self, question: str, examples: str, trajectory: str, reflections: str, depth: int, prompt: str, additional_keys: Dict[str, str]) -> tuple:
        pass

    @abstractmethod
    def generate_observation(self, key: str, action_type: str, query: str, trajectory: str, depth: int) -> tuple:
        pass

    @abstractmethod
    def select_node(self, node: Node) -> Node:
        pass

    @abstractmethod
    def expand_node(self, node: Node, question: str, key: str, examples: str, reflect_examples: str, prompt: str, reflect_prompt: str, additional_keys: Dict[str, str], reflect_additional_keys: Dict[str, str]) -> List[Node]:
        pass

    @abstractmethod
    def evaluate_node(self, node: Node, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> List[Dict]:
        pass

    @abstractmethod
    def simulate_node(self, node: Node, question: str, key: str, examples: str, reflect_examples: str, prompt: str, reflect_prompt: str, additional_keys: Dict[str, str], reflect_additional_keys: Dict[str, str]) -> tuple:
        pass

    @abstractmethod
    def backpropagate_node(self, node: Node, value: float) -> None:
        pass

    @abstractmethod
    def halting_condition(self, node: Node) -> bool:
        pass

    @abstractmethod
    def reflect_condition(self) -> bool:
        pass

    @abstractmethod
    def reflect(self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> List[Dict]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass