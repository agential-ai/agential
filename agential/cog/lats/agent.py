"""LATS agent.

Original Paper: https://arxiv.org/pdf/2310.04406
Paper Repository: https://github.com/lapisrocks/LanguageAgentTreeSearch
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.lats.factory import LATSFactory
from agential.cog.lats.functional import Node

class LATSAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        benchmark: str,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.benchmark = benchmark

        self.strategy = LATSFactory().get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            **strategy_kwargs,
        )
 
    def generate(
        self,
        question,
        max_iterations = 30
    ):
        node = self.strategy.initialize()
        for i in range(max_iterations):
            node = self.strategy.select_node(root)

            while node is None or (node.is_terminal and node.reward != 1):
                node = self.strategy.select_node(root)

            if node is None:
                break
    
            if node.is_terminal and node.reward == 1:
                return node.state, node.value, node.reward
            
            children_nodes, children_node_states = self.strategy.expand_node(
                node,

            )
        
            while node.is_terminal or not node.children:
                node = self.strategy.select_node(root)
                children_nodes, children_node_states = self.strategy.expand_node(node, args, task)

    def reset(self) -> Any:
        pass