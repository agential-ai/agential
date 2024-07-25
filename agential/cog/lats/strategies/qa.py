"""LATS Agent strategies for QA."""

import re

from typing import Dict, Tuple
from agential.eval.em import EM
from agential.cog.lats.strategies.base import LATSBaseStrategy
from agential.cog.lats.functional import (
    generate_prompt, 
    upward_traversal, 
    get_samples, 
    get_unique_trajectories,
    _prompt_reflection,
    _prompt_agent
)
from agential.cog.lats.functional import Node
from agential.cog.lats.prompts import HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT, LATS_REFLECT_INSTRUCTION_HOTPOTQA
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


class LATSQAStrategy(LATSBaseStrategy):

    def __init__(
        self, 
        llm, 
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        n_samples: int = 5,
        max_reflections: int = 4,
    ):
        super().__init__(llm)
        self.failed_trajectories = []
        self.docstore = docstore
        self.n_samples = n_samples
        self.max_reflections = max_reflections

    def generate(
        self,
        node,
        question,
        key,
        examples,
        reflect_examples,
        reflections,
        depth,
        prompt,
        reflect_prompt,
        additional_keys,
        reflect_additional_keys
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

        traversed_nodes = upward_traversal(node)
        trajectory = generate_prompt(traversed_nodes)

        unique_states = set()
        trajectories = []
        for _ in range(self.n_samples):
            trajectory_i, thought = self.generate_thought(
                question=question,
                examples=examples,
                trajectory=trajectory,
                reflections=reflections_str,
                depth=depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            trajectory_i, action_type, query = self.generate_action(
                question=question,
                examples=examples,
                trajectory=trajectory_i,
                reflections=reflections_str,
                depth=depth,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            unique_key = f"{thought}::{action_type}::{query}"
            if unique_key not in unique_states:
                unique_states.add(unique_key)
                
                trajectory_i, obs, reward, done, external_tool_info = self.generate_observation(
                    key=key,
                    action_type=action_type,
                    query=query,
                    trajectory=trajectory_i,
                    depth=depth,
                )
                trajectories.append(trajectory_i)

                new_node = Node(
                    state={
                        "thought": f"Thought {depth + 1}: {thought}",
                        "action": f"Action {depth + 1}: {action_type}[{query}]",
                        "observation": f"Observation {depth + 1}: {obs}",
                    },
                    parent=node,
                    depth=depth + 1,
                    is_terminal=reward==1 or done,
                    reward=reward
                )

                if new_node.is_terminal and reward == 0:
                    traversed_nodes = upward_traversal(new_node)
                    self.failed_trajectories.append({
                        "trajectory": traversed_nodes,
                        "final_answer": query.lower().strip()
                    })

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

    def select_node(self):
        pass

    def expand_node(self):
        pass

    def evaluate_node(self):
        pass

    def simulate_node(self):
        pass

    def backpropagate_node(self):
        pass

    def reflect_condition(self):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories)
        return len(unique_trajectories) > len(self.failed_trajectories) and len(unique_trajectories) < self.max_reflections

    def reflect(
        self, 
        question,
        examples,
        prompt, 
        additional_keys
    ):
        unique_trajectories = get_unique_trajectories(self.failed_trajectories)

        reflections = []
        for trajectory in unique_trajectories:
            reflection = _prompt_reflection(
                self.llm,
                question=question,
                examples=examples,
                trajectory=trajectory, 
                prompt=prompt, 
                additional_keys=additional_keys
            )

            reflections.append({
                'trajectory': trajectory,
                'reflection': reflection
            })

        return reflections

    def reset(self):
        pass
