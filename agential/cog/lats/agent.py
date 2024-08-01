"""LATS agent.

Original Paper: https://arxiv.org/pdf/2310.04406
Paper Repository: https://github.com/lapisrocks/LanguageAgentTreeSearch
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.lats.factory import LATSFactory


class LATSAgent(BaseAgent):
    """LATS (Language Agent Tree Search) agent.

    Attributes:
        llm: The language model used by the LATS agent.
        benchmark: The benchmark or task the agent is designed to solve.
    """
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
        patience=2, 
        max_iterations=30,
    ):
        root = self.strategy.initialize()
        for i in range(max_iterations):
            patience_counter, previous_node = 0, None
            node = self.strategy.select_node(root)

            while not node.children:
                # Early stopping if we cannot expand the tree.
                if node == previous_node:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                else:
                    patience_counter = 0
                previous_node = node

                node = self.strategy.select_node(root)
                children_nodes = self.strategy.expand_node(
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
                for child_node in children_nodes:
                    if self.strategy.halting_condition(child_node):
                        return child_node

            values = self.strategy.evaluate_node(
                node=node,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            reward, terminal_node, all_children_nodes, all_values = (
                self.strategy.simulate_node(
                    node=max(node.children, key=lambda child: child.value, default=node),
                    question=question,
                    key=key,
                    examples=examples,
                    reflect_examples=reflect_examples,
                    prompt=prompt,
                    reflect_prompt=reflect_prompt,
                    additional_keys=additional_keys,
                    reflect_additional_keys=reflect_additional_keys,
                )
            )

            if self.strategy.halting_condition(terminal_node):
                return terminal_node

            self.strategy.backpropagate_node(node=terminal_node, value=reward)

    def reset(self) -> None:
        self.strategy.reset()
