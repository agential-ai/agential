"""Functional module for Language Agent Tree Search (LATS)."""

from typing import Dict
import numpy as np

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.prompt import PromptTemplate


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

    def to_dict(self):
        return {
            "state": self.state,
            "parent": self.parent.to_dict() if self.parent else None,
            "children": [child.to_dict() for child in self.children],
            "visits": self.visits,
            "value": self.value,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "reward": self.reward,
        }

def _build_reflection_prompt(
    question: str,
    examples: str,
    trajectory: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    prompt = PromptTemplate.from_template(prompt).format(
        question=question, examples=examples, trajectory=trajectory, **additional_keys
    )
    return prompt


def _prompt_reflection(
    llm: BaseChatModel,
    question: str,
    examples: str,
    trajectory: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
):
    prompt = _build_reflection_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out


def _build_value_prompt(
    question: str,
    examples: str,
    trajectory: str,
    failed_trajectories: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
):
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        trajectory=trajectory,
        failed_trajectories=failed_trajectories,
        **additional_keys,
    )
    return prompt


def _prompt_value(
    llm: BaseChatModel,
    question: str,
    examples: str,
    trajectory: str,
    failed_trajectories: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
):
    prompt = _build_value_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        failed_trajectories=failed_trajectories,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out


def _build_agent_prompt(
    question: str,
    examples: str,
    trajectory: str,
    reflections: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
):
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        trajectory=trajectory,
        reflections=reflections,
        **additional_keys,
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    trajectory: str,
    reflections: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
):
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        reflections=reflections,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out


def get_node_trajectory(node):
    trajectory = []

    while node:
        step = []
        if node.depth > 0:
            if node.state.get("thought"):
                step.append(f"Thought {node.depth}: {node.state['thought']}")
            if node.state.get("action"):
                step.append(f"Action {node.depth}: {node.state['action']}")
            if node.state.get("observation"):
                step.append(f"Observation {node.depth}: {node.state['observation']}")
            step = "\n".join(step)
        trajectory.append(step)
        node = node.parent

    return "\n".join(reversed(trajectory))


def get_unique_trajectories(failed_trajectories, num=5):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get("final_answer")
        if final_answer not in seen_final_answers:
            unique_trajectories.append(traj["trajectory"])
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories