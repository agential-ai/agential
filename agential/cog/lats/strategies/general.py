"""LATS general strategy."""

import time

from typing import Any, Dict, List, Optional, Tuple

from agential.cog.lats.functional import (
    _prompt_agent,
    _prompt_reflection,
    accumulate_metrics,
    get_unique_trajectories,
)
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSOutput, LATSSimulationOutput, LATSStepOutput
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.llm.llm import BaseLLM, ModelResponse
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline


class LATSGeneralStrategy(LATSBaseStrategy):
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
    ) -> Any:
        start = time.time()

        if reset:
            self.reset()

        output = []

        root = self.initialize()
        for i in range(max_iterations):
            node = self.select_node(root)  # Selected node is always non-terminal.

            children_nodes, thought_model_responses, action_model_responses = (
                self.expand_node(
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
            )

            for child_node in children_nodes:
                if self.halting_condition(child_node):
                    output.append(
                        self.format_output(
                            iteration=i,
                            current_node=node,
                            children_nodes=children_nodes,
                            thought_model_responses=thought_model_responses,
                            action_model_responses=action_model_responses,
                            values=None,
                            values_responses=None,
                            simulation_reward=None,
                            simulation_terminal_node=None,
                            simulation_current_nodes=None,
                            simulation_children_nodes=None,
                            simulation_thought_model_responses=None,
                            simulation_action_model_responses=None,
                            simulation_values=None,
                            simulation_values_model_responses=None,
                        )
                    )
                    return child_node, output

            values, values_responses = self.evaluate_node(
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
                simulation_thought_model_responses,
                simulation_action_model_responses,
                simulation_values,
                simulation_values_model_responses,
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
                self.format_output(
                    iteration=i,
                    current_node=node,
                    children_nodes=children_nodes,
                    thought_model_responses=thought_model_responses,
                    action_model_responses=action_model_responses,
                    values=values,
                    values_responses=values_responses,
                    simulation_reward=simulation_reward,
                    simulation_terminal_node=simulation_terminal_node,
                    simulation_current_nodes=simulation_current_nodes,
                    simulation_children_nodes=simulation_children_nodes,
                    simulation_thought_model_responses=simulation_thought_model_responses,
                    simulation_action_model_responses=simulation_action_model_responses,
                    simulation_values=simulation_values,
                    simulation_values_model_responses=simulation_values_model_responses,
                )
            )

            if self.halting_condition(simulation_terminal_node):
                return simulation_terminal_node, output

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
    ) -> Tuple[List[Node], List[ModelResponse], List[ModelResponse]]:
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
            Tuple[List[Node], List[ModelResponse], List[ModelResponse]]: A list of generated child nodes, and the corresponding model responses.
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
    ) -> Tuple[str, str, ModelResponse]:
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
            Tuple[str, str, ModelResponse]: A tuple containing the updated trajectory, the generated thought, and the model response.
        """
        trajectory += f"\nThought {depth + 1}:"
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = out.choices[0].message.content

        thought = remove_newline(thought).split("Action")[0].strip()
        trajectory += " " + thought

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
    ) -> Tuple[str, str, str, ModelResponse]:
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
            Tuple[str, str, str, ModelResponse]: A tuple containing the updated trajectory, action type, query, and the model response.
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
    ) -> Tuple[List[Node], List[ModelResponse], List[ModelResponse]]:
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
            Tuple[List[Node], List[ModelResponse], List[ModelResponse]]: A list of generated child nodes, and the corresponding model responses.
        """
        if node.depth >= self.depth_limit:
            node.is_terminal = True
            return []
        children_nodes, thought_model_responses, action_model_responses = (
            self.generate_children_nodes(
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
        )
        node.add_children([node for node in children_nodes if node.parent])  # type: ignore

        return children_nodes, thought_model_responses, action_model_responses

    def evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], List[Optional[ModelResponse]]]:
        """Evaluate the given node and its children.

        Args:
            node (Node): The node to be evaluated.
            question (str): The main question or task.
            examples (str): Examples for context in evaluation.
            prompt (str): The prompt template for evaluation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[List[Dict[str, Any]], List[Optional[ModelResponse]]]: A list of dictionaries containing evaluation results for each child node and their model responses.
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
        List[List[ModelResponse]],
        List[List[ModelResponse]],
        List[List[Dict[str, Any]]],
        List[List[Optional[ModelResponse]]],
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
            Tuple[float, Node, List[Node], List[List[Node]], List[List[ModelResponse]], List[List[ModelResponse]], List[List[Dict[str, Any]]], List[List[Optional[ModelResponse]]]]:
                - The estimated value of the node.
                - The simulated node.
                - A list of the current nodes.
                - A list of the newly-created children nodes.
                - A list of thought model responses.
                - A list of action model responses.
                - A list of value estimates for newly-created children nodes.
                - A list of value model responses.
        """
        raise NotImplementedError

    def backpropagate_node(self, node: Node, value: float) -> None:
        """Backpropagate the estimated value through the tree, updating node statistics.

        Args:
            node (Node): The node from which to start backpropagation.
            value (float): The value to backpropagate through the tree.

        Returns:
            None
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
    ) -> List[Dict[str, str]]:
        """Perform reflection on the current search state.

        Args:
            question (str): The main question or task.
            examples (str): Examples for context in reflection.
            prompt (str): The prompt template for reflection.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing reflection results.
        """
        unique_trajectories = get_unique_trajectories(
            self.failed_trajectories, max_unique=self.max_unique
        )

        reflections: List[Dict[str, str]] = []
        for trajectory in unique_trajectories:
            reflection_out = _prompt_reflection(
                self.llm,
                question=question,
                examples=examples,
                trajectory=trajectory,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            reflection = reflection_out.choices[0].message.content

            reflections.append({"trajectory": trajectory, "reflection": reflection})

        self.reflection_map = reflections

        return reflections

    def format_output(
        self,
        iteration: int,
        current_node: Node,
        children_nodes: List[Node],
        thought_model_responses: List[ModelResponse],
        action_model_responses: List[ModelResponse],
        values: Optional[List[Dict[str, Any]]],
        values_responses: Optional[List[Optional[ModelResponse]]],
        simulation_reward: Optional[float],
        simulation_terminal_node: Optional[Node],
        simulation_current_nodes: Optional[List[Node]],
        simulation_children_nodes: Optional[List[List[Node]]],
        simulation_thought_model_responses: Optional[List[List[ModelResponse]]],
        simulation_action_model_responses: Optional[List[List[ModelResponse]]],
        simulation_values: Optional[List[List[Dict[str, Any]]]],
        simulation_values_model_responses: Optional[
            List[List[Optional[ModelResponse]]]
        ],
    ) -> Dict[str, Any]:
        if values_responses:
            values_metrics = [
                get_token_cost_time(response) if response is not None else None
                for response in values_responses
            ]
        else:
            values_metrics = []

        if simulation_current_nodes:
            simulation_current_nodes = [
                node.to_dict() for node in simulation_current_nodes
            ]
        else:
            simulation_current_nodes = []

        if simulation_children_nodes:
            simulation_children_nodes = [
                [child_node.to_dict() for child_node in child_nodes]
                for child_nodes in simulation_children_nodes
            ]
        else:
            simulation_children_nodes = []

        if simulation_thought_model_responses:
            simulation_thoughts_metrics = [
                [
                    get_token_cost_time(model_response)
                    for model_response in model_responses
                ]
                for model_responses in simulation_thought_model_responses
            ]
        else:
            simulation_thoughts_metrics = []

        if simulation_action_model_responses:
            simulation_actions_metrics = [
                [
                    get_token_cost_time(model_response)
                    for model_response in model_responses
                ]
                for model_responses in simulation_action_model_responses
            ]
        else:
            simulation_actions_metrics = []

        if simulation_values_model_responses:
            simulation_values_metrics = [
                [
                    get_token_cost_time(response) if response is not None else None
                    for response in simulation_values_model_response
                ]
                for simulation_values_model_response in simulation_values_model_responses
            ]
        else:
            simulation_values_metrics = []

        simulation_results = LATSSimulationOutput(
            simulation_reward=simulation_reward if simulation_reward else 0.0,
            simulation_terminal_node=(
                simulation_terminal_node.to_dict() if simulation_terminal_node else None
            ),
            simulation_current_nodes=simulation_current_nodes,
            simulation_children_nodes=simulation_children_nodes,
            simulation_thoughts_metrics=simulation_thoughts_metrics,
            simulation_actions_metrics=simulation_actions_metrics,
            simulation_values=simulation_values if simulation_values else [],
            simulation_values_metrics=simulation_values_metrics,
        )

        out = LATSStepOutput(
            iteration=iteration,
            current_node=current_node.to_dict(),
            children_nodes=[child_node.to_dict() for child_node in children_nodes],
            thoughts_metrics=[
                get_token_cost_time(response) for response in thought_model_responses
            ],
            actions_metrics=[
                get_token_cost_time(response) for response in action_model_responses
            ],
            values=values if values else [],
            values_metrics=values_metrics,
            simulation_results=simulation_results,
        )

        return out

    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None
