"""LATS Agent strategies for QA."""

import re

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
from agential.cog.lats.output import LATSSimulationOutput
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.react.output import ReActOutput
from agential.eval.em import EM
from agential.llm.llm import BaseLLM
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline
from agential.utils.general import get_token_cost_time

def get_node_trajectory_qa(node: Node) -> str:
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
                    f"Action {node.depth}: {node.state.action_type}[{node.state.query}]"
                )
            if node.state.observation:
                step.append(f"Observation {node.depth}: {node.state.observation}")
        step_str = "\n".join(step)
        trajectory.append(step_str)
        node = node.parent  # type: ignore

    return "\n".join(reversed(trajectory))


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


def parse_qa_value(string: str) -> Tuple[str, float]:
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


class LATSQAStrategy(LATSBaseStrategy):
    """A strategy class for QA benchmarks using the LATS agent.

    Attributes:
        llm: The language model to be used for generating responses.
        docstore (DocstoreExplorer): Document store explorer, defaults to Wikipedia.
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
        llm: BaseLLM,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        n_samples: int = 5,
        max_reflections: int = 4,
        depth_limit: int = 7,
        max_unique: int = 5,
        cache_values: bool = True,
    ) -> None:
        """Initialize."""
        super().__init__(llm)
        self.docstore = docstore
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
            "simulate_value": [],
            "reflection": []
        }

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
        is_simulate: bool
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

        trajectory = get_node_trajectory_qa(node)

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
                        answer="" if not done else query.lower().strip(),
                        external_tool_info=external_tool_info,
                    ),
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=reward == 1 or done,
                    reward=reward,
                )

                if new_node.is_terminal and reward == 0:
                    traversed_nodes = get_node_trajectory_qa(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": traversed_nodes,
                            "final_answer": query.lower().strip(),
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
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["thought"].append(get_token_cost_time(out))
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
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["action"].append(get_token_cost_time(out))
        action = out.choices[0].message.content

        action = remove_newline(action).split("Observation")[0]
        trajectory += " " + action
        action_type, query = parse_qa_action(action)

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
        external_tool_info = {"search_result": "", "lookup_result": ""}

        reward, done = 0, False
        trajectory += f"\nObservation {depth + 1}: "
        if action_type.lower() == "finish":
            if EM(query, key):
                obs = "Answer is CORRECT"
                reward = int(EM(query, key))
            else:
                obs = "Answer is INCORRECT"
            done = True
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
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
            is_simulate=False
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
            {"child_trajectory": get_node_trajectory_qa(child), "idx": idx}
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
                    value_str_out = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=examples,
                        trajectory=trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=prompt,
                        additional_keys=additional_keys,
                    )
                    self._prompt_metrics["value"].append(get_token_cost_time(value_str_out))
                    value_str = value_str_out.choices[0].message.content

                    if self.cache_values:
                        self.value_cache[unique_key] = value_str

                explanation, value = parse_qa_value(value_str)  # type: ignore
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
                is_simulate=True,
            )

            result["children_nodes"] = children_nodes

            for node in children_nodes:
                if node.is_terminal:
                    return node.reward, node, results

            for idx, child in enumerate(children_nodes):
                if not child.is_terminal:
                    child_trajectory = get_node_trajectory_qa(child)
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

                    value_str_out = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=value_examples,
                        trajectory=child_trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=value_prompt,
                        additional_keys=value_additional_keys,
                    )
                    self._prompt_metrics["simulate_value"].append(get_token_cost_time(value_str_out))

                    value_str = value_str_out.choices[0].message.content

                    explanation, value = parse_qa_value(value_str)  # type: ignore
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
            self._prompt_metrics["reflection"].append(get_token_cost_time(reflection_out))
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
        return {
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
        }

    def reset(self) -> None:
        """Reset the strategy to its initial state."""
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None


class LATSHotQAStrategy(LATSQAStrategy):
    """A strategy class for the HotpotQA benchmark using the LATS agent."""

    pass


class LATSTriviaQAStrategy(LATSQAStrategy):
    """A strategy class for the TriviaQA benchmark using the LATS agent."""

    pass


class LATSAmbigNQStrategy(LATSQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the LATS agent."""

    pass


class LATSFEVERStrategy(LATSQAStrategy):
    """A strategy class for the FEVER benchmark using the LATS agent."""

    pass
