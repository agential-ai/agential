"""LATS general strategy."""

import time

from typing import Any, Dict, List, Optional, Tuple

from agential.agent.lats.functional import (
    _prompt_agent,
    _prompt_reflection,
    accumulate_metrics,
    get_unique_trajectories,
)
from agential.agent.lats.node import Node
from agential.agent.lats.output import (
    LATSEvaluateResponse,
    LATSGenerateResponse,
    LATSOutput,
    LATSSimulationOutput,
    LATSSimulationResponse,
    LATSStepOutput,
)
from agential.agent.lats.strategies.base import LATSBaseStrategy
from agential.llm.llm import BaseLLM, Response
from agential.utils.parse import remove_newline


class LATSGeneralStrategy(LATSBaseStrategy):
    """LATS general strategy.

    Args:
        llm (BaseLLM): The LLM to use.
        n_samples (int): The number of samples to use. Defaults to 5.
        max_reflections (int): The maximum number of reflections to use. Defaults to 4.
        depth_limit (int): The maximum depth of the tree. Defaults to 7.
        max_unique (int): The maximum number of unique trajectories to use. Defaults to 5.
        cache_values (bool): Whether to cache values. Defaults to True.
        testing (bool): Whether to use testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        n_samples: int = 5,
        max_reflections: int = 4,
        depth_limit: int = 7,
        max_unique: int = 5,
        cache_values: bool = True,
        testing: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__(
            llm=llm,
            n_samples=n_samples,
            max_reflections=max_reflections,
            depth_limit=depth_limit,
            max_unique=max_unique,
            cache_values=cache_values,
            testing=testing,
        )

        self.failed_trajectories: List[Dict[str, str]] = []
        self.reflection_map: List[Dict[str, str]] = []
        self.value_cache: Dict[str, str] = {}
        self.root: Optional[Node] = None

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        value_examples: str,
        prompt: str,
        reflect_prompt: str,
        value_prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        value_additional_keys: Dict[str, str],
        max_iterations: int,
        reset: bool,
    ) -> LATSOutput:
        """Generate child nodes for the given node.

        Args:
            question (str): The question to answer.
            key (str): The key for the current node.
            examples (str): The examples for the current node.
            reflect_examples (str): The examples for the current node.
            value_examples (str): The examples for the current node.
            prompt (str): The prompt to use for the current node.
            reflect_prompt (str): The prompt to use for the current node.
            value_prompt (str): The prompt to use for the current node.
            additional_keys (Dict[str, str]): Additional keys for the current node.
            reflect_additional_keys (Dict[str, str]): Additional keys for the current node.
            value_additional_keys (Dict[str, str]): Additional keys for the current node.
            max_iterations (int): The maximum number of iterations.
            reset (bool): Whether to reset the strategy.

        Returns:
            LATSOutput: The output of the strategy.
        """
        start = time.time()

        if reset:
            self.reset()

        output = []

        root = self.initialize()
        for i in range(max_iterations):
            simulation_terminal_node = None
            node = self.select_node(root)  # Selected node is always non-terminal.

            (children_nodes, generate_response) = self.expand_node(
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
                if self.halting_condition(child_node):
                    output.append(
                        LATSStepOutput(
                            iteration=i,
                            current_node=node.to_dict(),
                            children_nodes=[node.to_dict() for node in children_nodes],
                            generate_response=generate_response,
                            values=None,
                            evaluate_response=None,
                            simulation_results=None,
                            simulation_response=None,
                        )
                    )
                    simulation_terminal_node = child_node
                    break

            if simulation_terminal_node:
                break

            values, evaluate_response = self.evaluate_node(
                node=node,
                question=question,
                examples=value_examples,
                prompt=value_prompt,
                additional_keys=value_additional_keys,
            )

            (
                simulation_reward,
                simulation_terminal_node,
                simulation_current_nodes,
                simulation_children_nodes,
                simulation_values,
                simulation_response,
            ) = self.simulate_node(
                node=max(node.children, key=lambda child: child.value, default=node),
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

            output.append(
                LATSStepOutput(
                    iteration=i,
                    current_node=node.to_dict(),
                    children_nodes=[node.to_dict() for node in children_nodes],
                    generate_response=generate_response,
                    values=values,
                    evaluate_response=evaluate_response,
                    simulation_results=LATSSimulationOutput(
                        simulation_reward=simulation_reward,
                        simulation_terminal_node=simulation_terminal_node.to_dict(),
                        simulation_current_nodes=[
                            node.to_dict() for node in simulation_current_nodes
                        ],
                        simulation_children_nodes=[
                            [node.to_dict() for node in children_nodes]
                            for children_nodes in simulation_children_nodes
                        ],
                        simulation_values=simulation_values,
                    ),
                    simulation_response=simulation_response,
                )
            )

            if self.halting_condition(simulation_terminal_node):
                break

            self.backpropagate_node(
                node=simulation_terminal_node, value=simulation_reward
            )

        total_time = time.time() - start
        total_metrics = accumulate_metrics(output)
        out = LATSOutput(
            answer=simulation_terminal_node,
            total_prompt_tokens=total_metrics["total_prompt_tokens"],
            total_completion_tokens=total_metrics["total_completion_tokens"],
            total_tokens=total_metrics["total_tokens"],
            total_prompt_cost=total_metrics["total_prompt_cost"],
            total_completion_cost=total_metrics["total_completion_cost"],
            total_cost=total_metrics["total_cost"],
            total_prompt_time=total_metrics["total_prompt_time"],
            total_time=total_time if not self.testing else 0.5,
            additional_info=output,
        )

        return out

    def initialize(self) -> Node:
        """Create and return the root node.

        Returns:
            Node: The root node of the search tree.
        """
        self.root = Node()  # type: ignore
        return self.root

    def generate_children_nodes(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
    ) -> Tuple[List[Node], LATSGenerateResponse]:
        """Generate child nodes for the given node.

        Args:
            node (Node): The current node to expand.
            question (str): The main question or task.
            key (str): The answer key for evaluation.
            examples (str): Examples for context.
            reflect_examples (str): Examples for reflection.
            prompt (str): The prompt template for generation.
            reflect_prompt (str): The prompt template for reflection.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            reflect_additional_keys (Dict[str, str]): Additional keys for reflection prompt formatting.

        Returns:
            Tuple[List[Node], LATSGenerateResponse]: A list of generated child nodes, and the pydantic of corresponding responses.
        """
        raise NotImplementedError

    def generate_thought(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generate a thought for the current step in the reasoning process.

        Args:
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for thought generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the thought process.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for thought generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, Response]: A tuple containing the updated trajectory, the generated thought, and the responses.
        """
        trajectory += f"\nThought {depth + 1}: "
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(out.output_text).split("Action")[0].strip()
        trajectory += thought

        return trajectory, thought, out

    def generate_action(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action for the current step in the reasoning process.

        Args:
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, Response]: A tuple containing the updated trajectory, action type, query, and the responses.
        """
        raise NotImplementedError

    def generate_observation(
        self,
        key: str,
        action_type: str,
        query: str,
        trajectory: str,
        depth: int,
    ) -> Tuple[str, int, str, bool, Dict[str, Any]]:
        """Generate an observation based on the current action.

        Args:
            key (str): The answer key for evaluation.
            action_type (str): The type of action taken.
            query (str): The query associated with the action.
            trajectory (str): The current trajectory or history of thoughts and actions.
            depth (int): The current depth in the search tree.

        Returns:
            Tuple[str, int, str, bool, Dict[str, str]]: A tuple containing the updated trajectory,
            reward, observation, done flag, and external tool information.
        """
        raise NotImplementedError

    def select_node(self, node: Node) -> Node:
        """Select the most promising node for expansion.

        There are 3 cases for the returned node:
            - Case 1 (Current node has no children): Returns current node as it has no children (root).
            - Case 2 (Backtracks till root): Returns current node as it has all terminal children (must be root).
            - Case 3 (Most common case): Returns non-terminal childless node with highest UCT value.

        Args:
            node (Node): The current node from which to start the selection.

        Returns:
            Node: The selected node for expansion.
        """
        while node and node.children:
            # Filter out terminal children.
            non_terminal_children = [
                child for child in node.children if not child.is_terminal
            ]

            # If all children are terminal, move up to the parent node.
            if not non_terminal_children:
                if node.parent:
                    node.parent.children.remove(node)
                    node = node.parent
                else:
                    # If we are at the root node and all children are terminal, return the root.
                    break
            else:
                # Select the child with the highest UCT value among non-terminal children.
                node = max(non_terminal_children, key=lambda child: child.uct())

        return node

    def expand_node(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
    ) -> Tuple[List[Node], LATSGenerateResponse]:
        """Expand the given node by generating its child nodes.

        Args:
            node (Node): The node to be expanded.
            question (str): The main question or task.
            key (str): The answer key for evaluation.
            examples (str): Examples for context in generation.
            reflect_examples (str): Examples for reflection.
            prompt (str): The prompt template for generation.
            reflect_prompt (str): The prompt template for reflection.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            reflect_additional_keys (Dict[str, str]): Additional keys for reflection prompt formatting.

        Returns:
            Tuple[List[Node], LATSGenerateResponse]: A list of generated child nodes, and the corresponding responses.
        """
        if node.depth >= self.depth_limit:
            node.is_terminal = True
            return [], LATSGenerateResponse(
                thoughts_response=[],
                actions_response=[],
                reflections_response=[],
            )

        children_nodes, generate_response = self.generate_children_nodes(
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
        node.add_children([node for node in children_nodes if node.parent])  # type: ignore

        return children_nodes, generate_response

    def evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], LATSEvaluateResponse]:
        """Evaluate the given node and its children.

        Args:
            node (Node): The node to be evaluated.
            question (str): The main question or task.
            examples (str): Examples for context in evaluation.
            prompt (str): The prompt template for evaluation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[List[Dict[str, Any]], LATSEvaluateResponse]: A list of dictionaries containing evaluation results for each child node and their responses.
        """
        raise NotImplementedError

    def simulate_node(
        self,
        node: Node,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        value_examples: str,
        prompt: str,
        reflect_prompt: str,
        value_prompt: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        value_additional_keys: Dict[str, str],
    ) -> Tuple[
        float,
        Node,
        List[Node],
        List[List[Node]],
        List[List[Dict[str, Any]]],
        LATSSimulationResponse,
    ]:
        """Simulate the node to estimate its value and collect information about the simulation process.

        Args:
            node (Node): The node to simulate.
            question (str): The main question or task.
            key (str): The answer key for evaluation.
            examples (str): Examples for context in simulation.
            reflect_examples (str): Examples for reflection during simulation.
            value_examples (str): Examples for value estimation.
            prompt (str): The prompt template for simulation.
            reflect_prompt (str): The prompt template for reflection during simulation.
            value_prompt (str): The prompt template for value estimation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            reflect_additional_keys (Dict[str, str]): Additional keys for reflection prompt formatting.
            value_additional_keys (Dict[str, str]): Additional keys for value estimation prompt formatting.

        Returns:
            Tuple[float, Node, List[Node], List[List[Node]], List[List[Dict[str, Any]]], LATSSimulationResponse]:
                - The estimated value of the node
                - The simulation's terminal node
                - Each simulation iteration's children nodes
                - Each simulation iteration's children nodes' values
                - Response for the simulation process
        """
        raise NotImplementedError

    def backpropagate_node(self, node: Node, value: float) -> None:
        """Backpropagate the estimated value through the tree, updating node statistics.

        Args:
            node (Node): The node from which to start backpropagation.
            value (float): The value to backpropagate through the tree.
        """
        while node:
            node.visits += 1
            if node.is_terminal:
                if node.reward == 0:
                    node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                else:
                    node.value = (node.value * (node.visits - 1) + value) / node.visits
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits

            node = node.parent  # type: ignore

    def halting_condition(self, node: Node) -> bool:
        """Determine if the search should halt at the current node.

        Args:
            node (Node): The current node to evaluate.

        Returns:
            bool: True if the search should halt, False otherwise.
        """
        return node.is_terminal and node.reward == 1

    def reflect_condition(self) -> bool:
        """Determine if reflection should be performed.

        Returns:
            bool: True if reflection should be performed, False otherwise.
        """
        unique_trajectories = get_unique_trajectories(
            self.failed_trajectories, max_unique=self.max_unique
        )
        return (
            len(unique_trajectories) > len(self.reflection_map)
            and len(unique_trajectories) < self.max_reflections
        )

    def reflect(
        self, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]
    ) -> Tuple[List[Dict[str, str]], List[Response]]:
        """Perform reflection on the current search state.

        Args:
            question (str): The main question or task.
            examples (str): Examples for context in reflection.
            prompt (str): The prompt template for reflection.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[List[Dict[str, str]], List[Response]]: A list of dictionaries containing reflection results and the responses.
        """
        unique_trajectories = get_unique_trajectories(
            self.failed_trajectories, max_unique=self.max_unique
        )

        reflections: List[Dict[str, str]] = []
        reflection_response: List[Response] = []
        for trajectory in unique_trajectories:
            reflection_out = _prompt_reflection(
                self.llm,
                question=question,
                examples=examples,
                trajectory=trajectory,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            reflection_response.append(reflection_out)
            reflections.append(
                {"trajectory": trajectory, "reflection": reflection_out.output_text}
            )

        self.reflection_map = reflections

        return reflections, reflection_response

    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None
