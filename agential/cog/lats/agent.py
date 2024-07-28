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
        key,
        examples,
        reflect_examples,
        prompt,
        reflect_prompt,
        additional_keys,
        reflect_additional_keys,
        max_iterations = 30
    ):
        root = self.strategy.initialize()
        for i in range(max_iterations):
            node = self.strategy.select_node(root)
    
            if self.strategy.halting_condition(node):
                return node
            
            children_nodes, children_node_states = self.strategy.expand_node(
                node=node,
                question=question,
                key=key,
                examples=examples,
                reflect_examples=reflect_examples,
                prompt=prompt,
                reflect_prompt=reflect_prompt,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
            )
        
            while node.is_terminal or not node.children:
                node = self.strategy.select_node(root)
                children_nodes, children_node_states = self.strategy.expand_node(
                    node=node,
                    question=question,
                    key=key,
                    examples=examples,
                    reflect_examples=reflect_examples,
                    prompt=prompt,
                    reflect_prompt=reflect_prompt,
                    additional_keys=additional_keys,
                    reflect_additional_keys=reflect_additional_keys,
                )

            values = self.strategy.evaluate_node(
                node=node,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            reward, terminal_node = self.strategy.simulate_node(
                node=max(node.children, key=lambda child: child.value),
                question=question,
                key=key,
                examples=examples,
                reflect_examples=reflect_examples,
                prompt=prompt,
                reflect_prompt=reflect_prompt,
                additional_keys=additional_keys,
                reflect_additional_keys=reflect_additional_keys,
            )

            if self.strategy.halting_condition(terminal_node):
                return terminal_node
            
            self.strategy.backpropagate_node(
                node=terminal_node,
                value=reward
            )

            

    def reset(self) -> Any:
        pass