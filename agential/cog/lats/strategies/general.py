"""LATS general strategy."""

from typing import Any, Dict, List, Optional, Tuple

from agential.cog.lats.functional import (
    _prompt_agent,
    _prompt_reflection,
    get_unique_trajectories,
)
from agential.cog.lats.node import Node
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.llm.llm import BaseLLM, ModelResponse
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
        if reset:
            self.reset()

        output = []

        root = self.initialize()
        for i in range(max_iterations):
            node = self.select_node(root)  # Selected node is always non-terminal.

            children_nodes = self.expand_node(
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
                        LATSOutput(
                            **self.create_output_dict(
                                iteration=i,
                                current_node=node,
                                children_nodes=children_nodes,
                                values=None,
                                simulation_reward=None,
                                simulation_terminal_node=None,
                                simulation_results=None,
                            )
                        )
                    )
                    return child_node, output

            values = self.evaluate_node(
                node=node,
                question=question,
                examples=value_examples,
                prompt=value_prompt,
                additional_keys=value_additional_keys,
            )

            simulation_reward, simulation_terminal_node, simulation_results = (
                self.simulate_node(
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

            output.append(
                LATSOutput(
                    **self.create_output_dict(
                        iteration=i,
                        current_node=node,
                        children_nodes=children_nodes,
                        values=values,
                        simulation_reward=simulation_reward,
                        simulation_terminal_node=simulation_terminal_node,
                        simulation_results=simulation_results,
                    )
                )
            )

            if self.halting_condition(simulation_terminal_node):
                return simulation_terminal_node, output

            self.backpropagate_node(
                node=simulation_terminal_node, value=simulation_reward
            )

        return simulation_terminal_node, output

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
            reflection = reflection_out.choices[0].message.content

            reflections.append({"trajectory": trajectory, "reflection": reflection})

        self.reflection_map = reflections

        return reflections

    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None
