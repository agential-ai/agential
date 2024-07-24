"""LATS Agent strategies for QA."""

from typing import Any
from agential.cog.lats.strategies.base import LATSBaseStrategy

class LATSQAStrategy(LATSBaseStrategy):

    def __init__(self, llm):
        super().__init__(llm)

    def generate(self):
        pass

    def select_node(self):
        pass

    def expand_node(self):
        pass

    def evaluate_node(self):
        pass

    def simulate_node(self):
        pass

    def backpropagate_node(self):
        pass

    def reflect_node(self):
        pass

    def reset(self):
        pass
