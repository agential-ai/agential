"""LATS Agent strategies for QA."""

import re

from typing import Tuple
from agential.eval.em import EM
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.functional import (
    get_node_trajectory,
    _prompt_value,
    get_unique_trajectories,
    _prompt_reflection,
    _prompt_agent,
)
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline
from agential.cog.lats.node import Node
from langchain_community.docstore.wikipedia import Wikipedia


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in LATS.

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


def parse_qa_value(string: str) -> Tuple[str, int]:
    try:
        explanation_part = string.split("Explanation:")[1].strip()
        explanation, score_part = explanation_part.split("Correctness score:")
        score = int(score_part.strip())
        return explanation.strip(), score
    except Exception:
        return "Explanation not found", 0


class LATSQAStrategy(LATSBaseStrategy):
    def __init__(
        self,
        llm,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        n_samples: int = 5,
        max_reflections: int = 4,
        depth_limit: int = 7,
        max_unique: int = 5,
        cache_values: bool = True,
    ):
        super().__init__(llm)
        self.docstore = docstore
        self.n_samples = n_samples
        self.max_reflections = max_reflections
        self.depth_limit = depth_limit
        self.max_unique = max_unique
        self.cache_values = cache_values

        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}
        self.root = None

    def initialize(self):
        self.root = Node()
        return self.root

    def generate(
        self,
        node: Node,
        question,
        key,
        examples,
        reflect_examples,
        prompt,
        reflect_prompt,
        additional_keys,
        reflect_additional_keys,
    ):
        reflections_str = ""
        if self.reflect_condition():
            reflections = self.reflect(
                question=question,
                examples=reflect_examples,
                prompt=reflect_prompt,
                additional_keys=reflect_additional_keys,
            )
            for reflection in reflections:
                reflections_str += f"{reflection['trajectory']}\nFAILED TRAJECTORY\nReflection: {reflection['reflection']}\n\n"

        trajectory = get_node_trajectory(node)

        unique_states = set()
        children_nodes = []
        children_node_states = []
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

                trajectory_i, reward, obs, done, external_tool_info = (
                    self.generate_observation(
                        key=key,
                        action_type=action_type,
                        query=query,
                        trajectory=trajectory_i,
                        depth=node.depth,
                    )
                )

                new_node = Node(
                    state={
                        "thought": f"Thought {node.depth + 1}: {thought}",
                        "action": f"Action {node.depth + 1}: {action_type}[{query}]",
                        "observation": f"Observation {node.depth + 1}: {obs}",
                    },
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=reward == 1 or done,
                    reward=reward,
                )

                if new_node.is_terminal and reward == 0:
                    traversed_nodes = get_node_trajectory(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": traversed_nodes,
                            "final_answer": query.lower().strip(),
                        }
                    )

                children_nodes.append(new_node)

                children_node_info = {
                    "thought": thought,
                    "action_type": action_type,
                    "query": query,
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "external_tool_info": external_tool_info,
                    "is_terminal": reward == 1 or done,
                    "depth": node.depth + 1,
                }

                children_node_states.append(children_node_info)

        return children_nodes, children_node_states

    def generate_thought(
        self,
        question,
        examples,
        trajectory,
        reflections,
        depth,
        prompt,
        additional_keys,
    ):
        trajectory += f"\nThought {depth + 1}:"
        thought = _prompt_agent(
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
        question,
        examples,
        trajectory,
        reflections,
        depth,
        prompt,
        additional_keys,
    ):
        trajectory += f"\nAction {depth + 1}:"
        action = _prompt_agent(
            question=question,
            examples=examples,
            trajectory=trajectory,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = remove_newline(action).split("Observation")[0]
        trajectory += " " + action
        action_type, query = parse_qa_action(action)

        return trajectory, action_type, query

    def generate_observation(
        self,
        key,
        action_type,
        query,
        trajectory,
        depth,
    ):
        external_tool_info = {"search_result": "", "lookup_result": ""}

        done = False
        trajectory += f"\nObservation {depth + 1}: "
        if action_type.lower() == "finish":
            if EM(query, key):
                obs = "Answer is CORRECT"
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

        return trajectory, int(EM(query, key)), obs, done, external_tool_info

    def select_node(self, node):
        while node and node.children:
            terminal_children = [child for child in node.children if child.is_terminal]

            if len(terminal_children) == len(node.children):
                if node.parent:
                    node.parent.children.remove(node)
                node = node.parent
                continue

            for child in terminal_children:
                if child.reward == 1:
                    return child

            node = max(
                [child for child in node.children if not child.is_terminal],
                key=lambda child: child.uct(),
            )

        return node

    def expand_node(
        self,
        node,
        question,
        key,
        examples,
        reflect_examples,
        prompt,
        reflect_prompt,
        additional_keys,
        reflect_additional_keys,
    ):
        if node.depth >= self.depth_limit:
            node.is_terminal = True
            return []
        children_nodes, children_node_states = self.generate(
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
        node.add_children(children_nodes)

        return children_nodes, children_node_states

    def evaluate_node(
        self,
        node,
        question,
        examples,
        prompt,
        additional_keys,
    ):
        children_trajectories = [
            {"child_trajectory": get_node_trajectory(child), "idx": idx}
            for idx, child in enumerate(node.children)
            if not child.is_terminal
        ]

        values = []
        child_trajectory_cache = {}
        for child_trajectory in children_trajectories:
            trajectory = child_trajectory["child_trajectory"]
            idx = child_trajectory["idx"]
            if trajectory in child_trajectory_cache:
                value = 0
            else:
                failed_trajectories = ""
                if len(self.reflection_map) > 0:
                    for trajectory_reflection in self.reflection_map:
                        failed_trajectories += f"Question: {question}\n{trajectory_reflection['trajectory']}\n\nExplanation: This trajectory is incorrect as {trajectory_reflection['reflection']}\nCorrectness score: 1"
                        failed_trajectories += "\n\n---\n\n"
                    failed_trajectories = (
                        failed_trajectories.strip().rstrip("---").strip()
                    )

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

                explanation, value = parse_qa_value(value_str)
                value = value / 10
                node.children[idx].value = value

                child_trajectory_cache[trajectory] = value
            values.append({"value": value, "explanation": explanation})

        return values

    def simulate_node(
        self,
        node,
        question,
        key,
        examples,
        reflect_examples,
        prompt,
        reflect_prompt,
        additional_keys,
        reflect_additional_keys,
    ):
        depth = node.depth
        rewards = [0]
        while not node.is_terminal and depth < self.depth_limit:
            values = []
            children_nodes, children_node_states = self.generate(
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

            for node in children_nodes:
                if node.is_terminal:
                    return node.reward, node

            for child in children_nodes:
                if not child.is_terminal:
                    child_trajectory = get_node_trajectory(child)
                    failed_trajectories = ""
                    if len(self.reflection_map) > 0:
                        for trajectory_reflection in self.reflection_map:
                            failed_trajectories += f"Question: {question}\n{trajectory_reflection['trajectory']}\n\nExplanation: This trajectory is incorrect as {trajectory_reflection['reflection']}\nCorrectness score: 1"
                            failed_trajectories += "\n\n---\n\n"
                        failed_trajectories = (
                            failed_trajectories.strip().rstrip("---").strip()
                        )
                    value = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=examples,
                        trajectory=child_trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=prompt,
                        additional_keys=additional_keys,
                    )

                    explanation, value = parse_qa_value(value)
                    values.append(value)

            max_value_index = values.index(max(values))
            rewards.append(max(values))
            node = children_nodes[max_value_index]
            depth += 1

            if depth == self.depth_limit:
                rewards = [-1]

        return sum(rewards) / len(rewards), node

    def backpropagate_node(self, node, value):
        while node:
            node.visits += 1
            if node.is_terminal:
                if node.reward == 0:
                    node.value = (node.value * (node.visits - 1) + (-1)) / node.visits
                else:
                    node.value = (node.value * (node.visits - 1) + value) / node.visits
            else:
                node.value = (node.value * (node.visits - 1) + value) / node.visits

            node = node.parent

    def halting_condition(self, node):
        return node.is_terminal and node.reward == 1

    def reflect_condition(self):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories, max_unique=self.max_unique)
        return (
            len(unique_trajectories) > len(self.reflection_map)
            and len(unique_trajectories) < self.max_reflections
        )

    def reflect(self, question, examples, prompt, additional_keys):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories, max_unique=self.max_unique)

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

    def reset(self):
        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}


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
