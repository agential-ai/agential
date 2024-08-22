"""Base LATS Agent strategy class."""

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from agential.cog.base.strategies import BaseStrategy
from agential.cog.lats.node import Node
from agential.cog.lats.output import (
    LATSEvaluateMetrics,
    LATSGenerateMetrics,
    LATSOutput,
    LATSSimulationMetrics,
)
from agential.llm.llm import BaseLLM
from agential.utils.metrics import Response


class LATSBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the LATS Agent."""

    def __init__(
        self,
        llm: BaseLLM,
        n_samples: int,
        max_reflections: int,
        depth_limit: int,
        max_unique: int,
        cache_values: bool,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.n_samples = n_samples
        self.max_reflections = max_reflections
        self.depth_limit = depth_limit
        self.max_unique = max_unique
        self.cache_values = cache_values

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def initialize(self) -> Node:
        """Create and return the rdeot node.

        Returns:
            Node: The root node of the search tree.
        """
        raise NotImplementedError

    @abstractmethod
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
    ) -> Tuple[List[Node], LATSGenerateMetrics]:
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
            Tuple[List[Node], LATSGenerateMetrics]: A list of generated child nodes, and the pydantic of corresponding metrics.
        """
        raise NotImplementedError

    @abstractmethod
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
            Tuple[str, str, Response]: A tuple containing the updated trajectory, the generated thought, and the metrics.
        """
        raise NotImplementedError

    @abstractmethod
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
            Tuple[str, str, str, Response]: A tuple containing the updated trajectory, action type, query, and the metrics.
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def select_node(self, node: Node) -> Node:
        """Select the most promising node for expansion.

        Args:
            node (Node): The current node from which to start the selection.

        Returns:
            Node: The selected node for expansion.
        """
        raise NotImplementedError

    @abstractmethod
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
    ) -> Tuple[List[Node], LATSGenerateMetrics]:
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
            Tuple[List[Node], LATSGenerateMetrics]: A list of generated child nodes, and the corresponding metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_node(
        self,
        node: Node,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], LATSEvaluateMetrics]:
        """Evaluate the given node and its children.

        Args:
            node (Node): The node to be evaluated.
            question (str): The main question or task.
            examples (str): Examples for context in evaluation.
            prompt (str): The prompt template for evaluation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[List[Dict[str, Any]], LATSEvaluateMetrics]: A list of dictionaries containing evaluation results for each child node and their metrics.
        """
        raise NotImplementedError

    @abstractmethod
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
        LATSSimulationMetrics,
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
            Tuple[float, Node, List[Node], List[List[Node]], List[List[Dict[str, Any]]], LATSSimulationMetrics]:
                - The estimated value of the node
                - The simulation's terminal node
                - Each simulation iteration's children nodes
                - Each simulation iteration's children nodes' values
                - Metrics for the simulation process
        """
        raise NotImplementedError

    @abstractmethod
    def backpropagate_node(self, node: Node, value: float) -> None:
        """Backpropagate the estimated value through the tree, updating node statistics.

        Args:
            node (Node): The node from which to start backpropagation.
            value (float): The value to backpropagate through the tree.
        """
        raise NotImplementedError

    @abstractmethod
    def halting_condition(self, node: Node) -> bool:
        """Determine if the search should halt at the current node.

        Args:
            node (Node): The current node to evaluate.

        Returns:
            bool: True if the search should halt, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reflect_condition(self) -> bool:
        """Determine if reflection should be performed.

        Returns:
            bool: True if reflection should be performed, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
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
            Tuple[List[Dict[str, str]], List[Response]]: A list of dictionaries containing reflection results and the metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        raise NotImplementedError
