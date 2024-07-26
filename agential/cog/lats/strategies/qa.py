"""LATS Agent strategies for QA."""

import re
import numpy as np

from typing import Tuple
from agential.eval.em import EM
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.functional import (
    generate_prompt,
    upward_traversal,
    _prompt_value,
    get_unique_trajectories,
    _prompt_reflection,
    _prompt_agent,
)
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline

from langchain_community.docstore.wikipedia import Wikipedia

def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct.

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
    
class Node:
    def __init__(
        self,
        state=None,
        parent=None,
        children=None,
        visits=0,
        value=0,
        depth=None,
        is_terminal=False,
        reward=0,
    ):
        self.state = (
            {"thought": "", "action": "", "observation": ""} if state is None else state
        )
        self.parent = parent
        self.children = [] if children is None else children
        self.visits = visits
        self.value = value
        self.depth = (
            0 if parent is None else parent.depth + 1 if depth is None else depth
        )
        self.is_terminal = is_terminal
        self.reward = reward

    def uct(self):
        if self.visits == 0:
            return self.value
        return self.value / self.visits + np.sqrt(
            2 * np.log(self.parent.visits) / self.visits
        )
    
    def add_children(self, children):
        self.children.extend(children)

class LATSQAStrategy(LATSBaseStrategy):

    def __init__(
        self,
        llm,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        n_samples: int = 5,
        max_reflections: int = 4,
        depth_limit: int = 7,
        cache_values: bool = True,
    ):
        super().__init__(llm)
        self.docstore = docstore
        self.n_samples = n_samples
        self.max_reflections = max_reflections
        self.depth_limit = depth_limit
        self.cache_values = cache_values

        self.failed_trajectories = []
        self.reflection_map = []
        self.value_cache = {}

    def generate(
        self,
        node: Node,
        question,
        key,
        examples,
        reflect_examples,
        reflections,
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

        # TODO: Every time we want to call upward traversal, we should need to traverse the whole tree again. Maybe we can just store this info in the node?
        traversed_nodes = upward_traversal(node)
        trajectory = generate_prompt(traversed_nodes)

        unique_states = set()
        children_nodes = []
        child_node_states = []
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

                trajectory_i, obs, reward, done, external_tool_info = (
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
                    traversed_nodes = upward_traversal(new_node)
                    self.failed_trajectories.append(
                        {
                            "trajectory": traversed_nodes,
                            "final_answer": query.lower().strip(),
                        }
                    )

                children_nodes.append(new_node)

                child_node_state = {
                    "thought": thought, 
                    "action_type": action_type,
                    "query": query, 
                    "obs": obs, 
                    "reward": reward, 
                    "done": done, 
                    "external_tool_info": external_tool_info        
                }

                child_node_states.append(child_node_state)

        return children_nodes, child_node_states

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

    # TODO: can we get this method to return less?
    def generate_observation(
        self,
        key,
        action_type,
        query,
        trajectory,
        depth,
    ):
        external_tool_info = {"search_result": "", "lookup_result": ""}

        reward = 0
        done = False
        trajectory += f"\nObservation {depth + 1}: "
        if action_type.lower() == "finish":
            if EM(query, key):
                obs = "Answer is CORRECT"
                reward = 1
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

        return trajectory, obs, reward, done, external_tool_info

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
                default=None,
            )

        return node

    def expand_node(
        self,
        node,
        question,
        key,
        examples,
        reflect_examples,
        reflections,
        prompt,
        reflect_prompt,
        additional_keys,
        reflect_additional_keys,
    ):
        if node.depth >= self.depth_limit:
            node.is_terminal = True
            return []
        children_nodes = self.generate(
            node=node,
            question=question,
            key=key,
            examples=examples,
            reflect_examples=reflect_examples,
            reflections=reflections,
            prompt=prompt,
            reflect_prompt=reflect_prompt,
            additional_keys=additional_keys,
            reflect_additional_keys=reflect_additional_keys,
        )
        node.add_children(children_nodes)

        return children_nodes

    # TODO: maybe we need structured outputs here too...
    def evaluate_node(
        self,
        node,
        question,
        examples,
        prompt,
        additional_keys,
    ):
        children_trajectories = [
            generate_prompt(upward_traversal(child))
            for child in node.children
            if not child.is_terminal
        ]

        values = []
        child_trajectory_cache = {}
        for child_trajectory in children_trajectories:
            if child_trajectory in child_trajectory_cache:
                value = 0
            else:
                failed_trajectories = ""
                if len(self.reflection_map) > 0:
                    for trajectory_reflection in self.reflection_map:
                        failed_trajectories += f"Question: {question}\n{trajectory_reflection['trajectory']}\n\nExplanation: This trajectory is incorrect as {trajectory_reflection['reflection']}\nCorrectness score: 1"
                        failed_trajectories += "\n\n---\n\n"
                    failed_trajectories = failed_trajectories.strip().rstrip("---").strip()

                unique_key = f"{child_trajectory}::{failed_trajectories}"
                if self.cache_values and unique_key in self.value_cache:
                    value_str = self.value_cache[unique_key]
                else:
                    value_str = _prompt_value(
                        llm=self.llm,
                        question=question,
                        examples=examples,
                        trajectory=child_trajectory,
                        failed_trajectories=failed_trajectories,
                        prompt=prompt,
                        additional_keys=additional_keys,
                    )

                    if self.cache_values:
                        self.value_cache[unique_key] = value_str
                
                explanation, value = parse_qa_value(value_str)
                value = value / 10

                child_trajectory_cache[child_trajectory] = value
            values.append(value)

        return values

    def simulate_node(self):
        pass

    def backpropagate_node(self):
        pass

    def reflect_condition(self):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories)
        return (
            len(unique_trajectories) > len(self.reflection_map)
            and len(unique_trajectories) < self.max_reflections
        )

    def reflect(self, question, examples, prompt, additional_keys):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories)

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
        pass
