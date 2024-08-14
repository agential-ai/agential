"""LATS general strategy."""

import re

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.lats.functional import (
    _build_failed_trajectory_format,
    _build_reflection_format,
    _prompt_agent,
    _prompt_reflection,
    _prompt_value,
    get_unique_trajectories,
)
from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSReActOutput, LATSSimulationOutput
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.eval.em import EM
from agential.llm.llm import BaseLLM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline

class LATSGeneralStrategy(LATSBaseStrategy):
    """A general strategy class for LATS agent."""

    def __init__(
        self,
        llm: BaseLLM,
        n_samples: int = 5,
        max_reflections: int = 4,
        depth_limit: int = 7,
        max_unique: int = 5,
        cache_values: bool = True,
    ) -> None:
        """Initialize."""
        super().__init__(llm)
        self.n_samples = n_samples
        self.max_reflections = max_reflections
        self.depth_limit = depth_limit
        self.max_unique = max_unique
        self.cache_values = cache_values

        self.failed_trajectories: List[Dict[str, str]] = []
        self.reflection_map: List[Dict[str, str]] = []
        self.value_cache: Dict[str, str] = {}
        self.root: Optional[Node] = None
        self._prompt_metrics: Dict[str, Any] = {
            "thought": [],
            "action": [],
            "value": [],
            "simulate_thought": [],
            "simulate_action": [],
            "simulate_value": [],
            "reflection": [],
        }


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
        is_simulate: bool,
    ) -> List[Node]:
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
            is_simulate (bool): Whether this method is called to simulate expansion or not.

        Returns:
            List[Node]: A list of generated child nodes.
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
        is_simulate: bool,
    ) -> Tuple[str, str]:
        """Generate a thought for the current step in the reasoning process.

        Args:
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for thought generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the thought process.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for thought generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            is_simulate (bool): Whether this method is called to simulate expansion or not.

        Returns:
            Tuple[str, str]: A tuple containing the updated trajectory and the generated thought.
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
        metric_key = "simulate_thought" if is_simulate else "thought"
        self._prompt_metrics[metric_key].append(get_token_cost_time(out))
        thought = out.choices[0].message.content

        thought = remove_newline(thought).split("Action")[0].strip()
        trajectory += " " + thought

        return trajectory, thought
    

    def generate_action(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
        is_simulate: bool,
    ) -> Tuple[str, str, str]:
        """Generate an action for the current step in the reasoning process.

        Args:
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.
            is_simulate (bool): Whether this method is called to simulate expansion or not.

        Returns:
            Tuple[str, str, str]: A tuple containing the updated trajectory, action type, and query.
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
    ) -> List[Node]:
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
            List[Node]: A list of newly generated child nodes.
        """
        if node.depth >= self.depth_limit:
            node.is_terminal = True
            return []
        children_nodes = self.generate(
            node=node,
            question=question,
            key=key,
            examples=examples,
            reflect_examples=reflect_examples,
            prompt=prompt,
            reflect_prompt=reflect_prompt,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
            is_simulate=False,
        )
        node.add_children(children_nodes)  # type: ignore

        return children_nodes
    

    def evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Evaluate the given node and its children.

        Args:
            node (Node): The node to be evaluated.
            question (str): The main question or task.
            examples (str): Examples for context in evaluation.
            prompt (str): The prompt template for evaluation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each child node.
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
    ) -> Tuple[float, Node, List[Dict[str, Any]]]:
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
            Tuple[float, Node, List[Dict[str, Any]]]: A tuple containing:
                - The estimated value of the node (float)
                - The final node reached in the simulation (Node)
                - A list of dictionaries, representing the states of nodes explored during simulation
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
            self._prompt_metrics["reflection"].append(
                get_token_cost_time(reflection_out)
            )
            reflection = reflection_out.choices[0].message.content

            reflections.append({"trajectory": trajectory, "reflection": reflection})

        self.reflection_map = reflections

        return reflections

    def create_output_dict(
        self,
        iteration: int,
        current_node: Node,
        children_nodes: List[Node],
        values: Optional[List[Dict[str, Any]]],
        simulation_reward: Optional[float],
        simulation_terminal_node: Optional[Node],
        simulation_results: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Create a dictionary containing the output of a LATS iteration.

        Args:
            iteration (int): The current iteration number.
            current_node (Node): The current node being processed.
            children_nodes (List[Node]): List of child nodes of the current node.
            values (Optional[List[Dict[str, Any]]]): List of values associated with the children nodes.
            simulation_reward (Optional[float]): The reward obtained from the simulation.
            simulation_terminal_node (Optional[Node]): The terminal node reached in the simulation.
            simulation_results (Optional[List[Dict[str, Any]]]): Results from multiple simulations.

        Returns:
            Dict[str, Any]: A dictionary containing the processed output of the LATS iteration,
            including the current state, children nodes, values, simulation results, and other
            relevant information.
        """
        if simulation_results:
            simulation_results_output = [
                LATSSimulationOutput(
                    current_node=result["current_node"].to_dict(),
                    children_nodes=[
                        child_node.to_dict() for child_node in result["children_nodes"]
                    ],
                    values=result["values"],
                )
                for result in simulation_results
            ]
        out = {
            "iteration": iteration,
            "current_node": current_node.to_dict(),
            "children_nodes": [child_node.to_dict() for child_node in children_nodes],
            "values": values if values else [],
            "simulation_reward": simulation_reward if simulation_reward else 0,
            "simulation_terminal_node": (
                simulation_terminal_node.to_dict() if simulation_terminal_node else {}
            ),
            "simulation_results": (
                simulation_results_output if simulation_results else []
            ),
            "prompt_metrics": deepcopy(self._prompt_metrics),
        }
        self._prompt_metrics = {
            "thought": [],
            "action": [],
            "value": [],
            "simulate_thought": [],
            "simulate_action": [],
            "simulate_value": [],
            "reflection": [],
        }
        return out

    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None
        self._prompt_metrics = {
            "thought": [],
            "action": [],
            "value": [],
            "simulate_thought": [],
            "simulate_action": [],
            "simulate_value": [],
            "reflection": [],
        }