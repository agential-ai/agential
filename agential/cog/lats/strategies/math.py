"""LATS Agent strategies for Math."""

import re

from typing import Any, Dict, List, Optional, Tuple

from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.lats.functional import (
    _build_failed_trajectory_format,
    _build_reflection_format,
    _prompt_agent,
    _prompt_reflection,
    _prompt_value,
    get_unique_trajectories,
)
from agential.cog.lats.node import Node
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.react.output import ReActOutput
from agential.eval.em import EM
from agential.utils.general import safe_execute
from agential.utils.parse import remove_newline


def get_node_trajectory_math(node: Node) -> str:
    """Generates a string representation of the trajectory from the given node to the root.

    Args:
        node (Node): The current node in the tree.

    Returns:
        str: A string representation of the trajectory, including thoughts, actions, and observations.
    """
    trajectory = []

    while node:
        step = []
        if node.depth > 0:
            if node.state.thought:
                step.append(f"Thought {node.depth}: {node.state.thought}")
            if node.state.action_type and node.state.query:
                step.append(
                    f"Action {node.depth}: {node.state.action_type}[\n```python\n{node.state.query}\n```\n]"
                )
            if node.state.observation:
                step.append(f"Observation {node.depth}: {node.state.observation}")
        step_str = "\n".join(step)
        trajectory.append(step_str)
        node = node.parent  # type: ignore

    return "\n".join(reversed(trajectory))


def parse_math_action(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`, `Calculate`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    match = re.search(r"\b(Finish|Calculate)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


def parse_math_value(string: str) -> Tuple[str, float]:
    """Extracts the explanation and correctness score from a given string.

    Args:
        string (str): The input string containing an explanation and correctness score.

    Returns:
        Tuple[str, float]: A tuple containing the explanation (str) and the correctness score (float).
        If parsing fails, returns ("Explanation not found", 0.0).
    """
    try:
        explanation_part = string.split("Explanation:")[1].strip()
        explanation, score_part = explanation_part.split("Correctness score:")
        score = float(int(score_part.strip()))
        return explanation.strip(), score
    except Exception:
        return "Explanation not found", 0.0


class LATSMathStrategy(LATSBaseStrategy):
    """A strategy class for Math benchmarks using the LATS agent.

    Attributes:
        llm: The language model to be used for generating responses.
        n_samples (int): Number of samples to generate, default is 5.
        max_reflections (int): Maximum number of reflections allowed, default is 4.
        depth_limit (int): Maximum depth of the search tree, default is 7.
        max_unique (int): Maximum number of unique samples to consider, default is 5.
        cache_values (bool): Whether to cache values, default is True.

    The strategy uses these parameters to fine-tune its behavior and performance
    in question-answering tasks.
    """

    def __init__(
        self,
        llm: BaseChatModel,
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

    def initialize(self) -> Node:
        """Create and return the root node.

        Returns:
            Node: The root node of the search tree.
        """
        self.root = Node()  # type: ignore
        return self.root

    def generate(
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
            List[Node]: A list of generated child nodes.
        """
        reflections_str = ""
        if self.reflect_condition():
            reflections = self.reflect(
                question=question,
                examples=reflect_examples,
                prompt=reflect_prompt,
                additional_keys=reflect_additional_keys,
            )
            for reflection in reflections:
                reflections_str += (
                    _build_reflection_format(
                        trajectory=reflection["trajectory"],
                        reflection=reflection["reflection"],
                    )
                    + "\n\n"
                )

        trajectory = get_node_trajectory_math(node)

        unique_states = set()
        children_nodes = []
        for _ in range(self.n_samples):
            trajectory_i, thought = self.generate_thought(
                question=question,
                examples=examples,
                trajectory=trajectory,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            trajectory_i, action_type, query = self.generate_action(
                question=question,
                examples=examples,
                trajectory=trajectory_i,
                reflections=reflections_str,
                depth=node.depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            unique_key = f"{thought}::{action_type}::{query}"
            if unique_key not in unique_states:
                unique_states.add(unique_key)

                _, reward, obs, done, external_tool_info = self.generate_observation(
                    key=key,
                    action_type=action_type,
                    query=query,
                    trajectory=trajectory_i,
                    depth=node.depth,
                )

                new_node = Node(
                    state=ReActOutput(
                        thought=thought,
                        action_type=action_type,
                        query=query,
                        observation=obs,
                        answer="" if not done else query,
                        external_tool_info=external_tool_info,
                    ),
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=reward == 1 or done,
                    reward=reward,
                )

                if new_node.is_terminal and reward == 0:
                    traversed_nodes = get_node_trajectory_math(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": traversed_nodes,
                            "final_answer": query,
                        }
                    )

                children_nodes.append(new_node)

        return children_nodes

    def generate_thought(
        self,
        question: str,
        examples: str,
        trajectory: str,
        reflections: str,
        depth: int,
        prompt: str,
        additional_keys: Dict[str, str],
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

        Returns:
            Tuple[str, str]: A tuple containing the updated trajectory and the generated thought.
        """
        trajectory += f"\nThought {depth + 1}:"
        thought = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
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

        Returns:
            Tuple[str, str, str]: A tuple containing the updated trajectory, action type, and query.
        """
        trajectory += f"\nAction {depth + 1}:"
        action = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_action(action)
        trajectory += f" {action_type}[\n```python\n{query}\n```\n]"

        return trajectory, action_type, query

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
        external_tool_info = {"execution_status": "", "code_answer": ""}
        code_answer, execution_status = safe_execute(query)

        reward, done = 0, False
        trajectory += f"\nObservation {depth + 1}: "
        if action_type.lower() == "finish":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            if EM(code_answer[0], key, normalize=False):
                obs = "Answer is CORRECT"
                reward = int(EM(code_answer[0], key, normalize=False))
            else:
                obs = "Answer is INCORRECT"
            done = True
        elif action_type.lower() == "calculate":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            obs = f"\n```python\n{query}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = (
                "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
            )
        trajectory += obs

        return trajectory, reward, obs, done, external_tool_info

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
        children_trajectories = [
            {"child_trajectory": get_node_trajectory_math(child), "idx": idx}
            for idx, child in enumerate(node.children)
            if not child.is_terminal
        ]

        values = []
        child_trajectory_cache = {}
        for child_trajectory in children_trajectories:
            trajectory: str = child_trajectory["child_trajectory"]  # type: ignore
            idx: int = child_trajectory["idx"]  # type: ignore
            if trajectory in child_trajectory_cache:
                value = 0
            else:
                failed_trajectories = ""
                if len(self.reflection_map) > 0:
                    for trajectory_reflection in self.reflection_map:
                        failed_trajectories += (
                            _build_failed_trajectory_format(
                                question=question,
                                trajectory=trajectory_reflection["trajectory"],
                                reflection=trajectory_reflection["reflection"],
                            )
                            + "\n\n"
                        )
                    failed_trajectories = failed_trajectories.rstrip("\n\n")

                unique_key = f"{trajectory}::{failed_trajectories}"
                if self.cache_values and unique_key in self.value_cache:
                    value_str = self.value_cache[unique_key]
                else:
                    value_str = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=examples,
                        trajectory=trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=prompt,
                        additional_keys=additional_keys,
                    )

                    if self.cache_values:
                        self.value_cache[unique_key] = value_str

                explanation, value = parse_math_value(value_str)  # type: ignore
                value = value / 10.0  # type: ignore
                node.children[idx].value = value

                child_trajectory_cache[trajectory] = value
            values.append({"node_idx": idx, "explanation": explanation, "value": value})

        return values

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
        depth = node.depth
        rewards: List[int] = [0]
        results: List[Dict[str, Any]] = []
        while not node.is_terminal and depth < self.depth_limit:
            result = {
                "current_node": node,
                "children_nodes": [],
                "values": [],
            }

            values: List[Dict[str, Any]] = []
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
            )

            result["children_nodes"] = children_nodes

            for node in children_nodes:
                if node.is_terminal:
                    return node.reward, node, results

            for idx, child in enumerate(children_nodes):
                if not child.is_terminal:
                    child_trajectory = get_node_trajectory_math(child)
                    failed_trajectories = ""
                    if len(self.reflection_map) > 0:
                        for trajectory_reflection in self.reflection_map:
                            failed_trajectories += (
                                _build_failed_trajectory_format(
                                    question=question,
                                    trajectory=trajectory_reflection["trajectory"],
                                    reflection=trajectory_reflection["reflection"],
                                )
                                + "\n\n"
                            )
                        failed_trajectories = failed_trajectories.rstrip("\n\n")

                    value = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=value_examples,
                        trajectory=child_trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=value_prompt,
                        additional_keys=value_additional_keys,
                    )

                    explanation, value = parse_math_value(value)  # type: ignore
                    values.append(
                        {"node_idx": idx, "explanation": explanation, "value": value}
                    )

            max_value = max(values, key=lambda x: x["value"])  # type: ignore
            max_value_index = values.index(max_value)
            rewards.append(max_value)  # type: ignore
            node = children_nodes[max_value_index]
            depth += 1

            if depth == self.depth_limit:
                rewards = [-1]

            result["best_child_node"] = node
            result["values"] = values

            results.append(result)

        return sum(rewards) / len(rewards), node, results

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

        reflections = []
        for trajectory in unique_trajectories:
            reflection = _prompt_reflection(
                self.llm,
                question=question,
                examples=examples,
                trajectory=trajectory,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            reflections.append({"trajectory": trajectory, "reflection": reflection})

        self.reflection_map = reflections

        return reflections

    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None


class LATSGSM8KStrategy(LATSMathStrategy):
    """A strategy class for the GSM8K benchmark using the LATS agent."""

    pass


class LATSSVAMPStrategy(LATSMathStrategy):
    """A strategy class for the SVAMP benchmark using the LATS agent."""

    pass


class LATSTabMWPStrategy(LATSMathStrategy):
    """A strategy class for the TabMWP benchmark using the LATS agent."""

    pass
