"""LATS agent.

Original Paper: https://arxiv.org/pdf/2310.04406
Paper Repository: https://github.com/lapisrocks/LanguageAgentTreeSearch
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.lats.factory import LATSFactory
from agential.cog.lats.output import LATSOutput, LATSSimulationOutput


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
        value_examples,
        prompt,
        reflect_prompt,
        value_prompt,
        additional_keys,
        reflect_additional_keys,
        value_additional_keys,
        max_iterations=30,
        reset=True,
    ):
        """Generate an output for the given question.

        Args:
            question (str): The question or task to be solved.
            key (str): The key associated with the question.
            examples (str): Examples to guide the agent's reasoning.
            reflect_examples (str): Examples to guide the agent's reflections.
            value_examples (str): Examples to guide the agent's value estimation.
            prompt (str): The prompt to guide the agent's reasoning.
            reflect_prompt (str): The prompt to guide the agent's reflections.
            value_prompt (str): The prompt to guide the agent's value estimation.
            additional_keys (Dict[str, str]): Additional keys for formatting the prompts.
            reflect_additional_keys (Dict[str, str]): Additional keys for formatting the reflection prompts.
            value_additional_keys (Dict[str, str]): Additional keys for formatting the value prompts.
            max_iterations (int): The maximum number of iterations to run the agent. Defaults to 30.
            reset (bool): Whether to reset the agent before generating the output. Defaults to True.

        Returns:
                Tuple[Node, List[LATSOutput]]: A tuple containing the root node and a list of outputs.
        """
        if reset:
            self.reset()

        output = []

        root = self.strategy.initialize()
        for i in range(max_iterations):
            node = self.strategy.select_node(
                root
            )  # Selected node is always non-terminal.

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
            print(
                "<===========================================================EXPAND NODE END===========================================================>"
            )

            for child_node in children_nodes:
                if self.strategy.halting_condition(child_node):
                    output.append(
                        LATSOutput(
                            iteration=i,
                            current_node=node.to_dict(),
                            children_nodes=[
                                child_node.to_dict() for child_node in children_nodes
                            ],
                            values=[],
                            simulation_reward=0,
                            simulation_terminal_node={},
                            simulation_results=[],
                        )
                    )
                    return child_node, output

            values = self.strategy.evaluate_node(
                node=node,
                question=question,
                examples=value_examples,
                prompt=value_prompt,
                additional_keys=value_additional_keys,
            )
            print(
                "<===========================================================EVALUATE NODE END===========================================================>"
            )

            simulation_reward, simulation_terminal_node, simulation_results = (
                self.strategy.simulate_node(
                    node=max(
                        node.children, key=lambda child: child.value, default=node
                    ),
                    question=question,
                    key=key,
                    examples=examples,
                    reflect_examples=reflect_examples,
                    value_examples=value_examples,
                    prompt=prompt,
                    reflect_prompt=reflect_prompt,
                    value_prompt=value_prompt,
                    additional_keys=additional_keys,
                    reflect_additional_keys=reflect_additional_keys,
                    value_additional_keys=value_additional_keys,
                )
            )
            print(
                "<===========================================================SIMULATE NODE END===========================================================>"
            )

            simulation_results = [
                LATSSimulationOutput(
                    current_node=result["current_node"].to_dict(),
                    children_nodes=[
                        child_node.to_dict() for child_node in result["children_nodes"]
                    ],
                    values=result["values"],
                )
                for result in simulation_results
            ]
            output.append(
                LATSOutput(
                    iteration=i,
                    current_node=node.to_dict(),
                    children_nodes=[
                        child_node.to_dict() for child_node in children_nodes
                    ],
                    values=values,
                    simulation_reward=simulation_reward,
                    simulation_terminal_node=simulation_terminal_node.to_dict(),
                    simulation_results=simulation_results,
                )
            )

            if self.strategy.halting_condition(simulation_terminal_node):
                return simulation_terminal_node, output

            self.strategy.backpropagate_node(
                node=simulation_terminal_node, value=simulation_reward
            )

        return simulation_terminal_node, output

    def reset(self) -> None:
        """Reset the agent."""
        self.strategy.reset()
