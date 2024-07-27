"""Unit tests for LATS functional module."""

import pytest


from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.lats.functional import (
    Node,
    get_node_trajectory,
    get_unique_trajectories,
    _build_reflection_prompt,
    _prompt_reflection,
    _build_value_prompt,
    _build_agent_prompt,
    _prompt_value,
    _prompt_agent,
)


from agential.cog.react.prompts import REACT_INSTRUCTION_HOTPOTQA
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT


@pytest.fixture
def sample_node():
    return Node(
        state={
            "thought": "Initial thought",
            "action": "Search[term]",
            "observation": "Result found",
        }
    )


@pytest.fixture
def sample_nodes():
    root = Node(state={"thought": "Root thought", "action": "", "observation": ""})
    child1 = Node(
        state={
            "thought": "Child1 thought",
            "action": "Lookup[topic]",
            "observation": "",
        },
        parent=root,
    )
    child2 = Node(
        state={
            "thought": "Child2 thought",
            "action": "Finish[answer]",
            "observation": "Answer correct",
        },
        parent=root,
        is_terminal=True,
        reward=1,
    )
    root.children = [child1, child2]
    return root, child1, child2


def test_get_node_trajectory(sample_nodes):
    root, child1, _ = sample_nodes
    expected_trajectory = (
        "Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]"
    )
    assert get_node_trajectory(child1) == expected_trajectory


def test_get_unique_trajectories():
    failed_trajectories = [
        {"trajectory": "Path1", "final_answer": "Answer1"},
        {"trajectory": "Path2", "final_answer": "Answer1"},  # Duplicate answer.
        {"trajectory": "Path3", "final_answer": "Answer2"},
    ]
    assert get_unique_trajectories(failed_trajectories) == ["Path1", "Path3"]



def test__build_reflection_prompt():
    """Test _build_reflection_prompt function."""
    prompt = _build_reflection_prompt(
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(prompt, str)

    # Test with custom prompt template string.
    gt_out = "  examples 1"  
    out = _build_reflection_prompt(
        question="",
        scratchpad="",
        examples="examples",
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert out == gt_out

def test__prompt_reflection():
    """Test _prompt_reflection function."""
    out = _prompt_reflection(
        llm=FakeListChatModel(responses=["Reflection Output"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Reflection Output"

    # Test with custom prompt template string.
    out = _prompt_reflection(
        llm=FakeListChatModel(responses=["Reflection Output"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert isinstance(out, str)
    assert out == "Reflection Output"

def test__build_value_prompt():
    """Test _build_value_prompt function."""
    prompt = _build_value_prompt(
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        failed_trajectories="Failed Trajectories",
    )
    assert isinstance(prompt, str)

    # Test with custom prompt template string.
    gt_out = "  examples Failed Trajectories 1"
    out = _build_value_prompt(
        question="",
        scratchpad="",
        examples="examples",
        failed_trajectories="Failed Trajectories",
        prompt="{question} {scratchpad} {examples} {failed_trajectories} {max_steps}",
    )
    assert out == gt_out

def test__prompt_value():
    """Test _prompt_value function."""
    out = _prompt_value(
        llm=FakeListChatModel(responses=["Value Output"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        failed_trajectories="Failed Trajectories",
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Value Output"

    # Test with custom prompt template string.
    out = _prompt_value(
        llm=FakeListChatModel(responses=["Value Output"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        failed_trajectories="Failed Trajectories",
        prompt="{question} {scratchpad} {examples} {failed_trajectories} {max_steps}",
    )
    assert isinstance(out, str)
    assert out == "Value Output"


def test__build_agent_prompt():
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        reflections="Reflections",
    )
    assert isinstance(prompt, str)

    # Test with custom prompt template string.
    gt_out = "  examples Reflections 1"
    out = _build_agent_prompt(
        question="",
        scratchpad="",
        examples="examples",
        reflections="Reflections",
        prompt="{question} {scratchpad} {examples} {reflections} {max_steps}",
    )
    assert out == gt_out

def test__prompt_agent():
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["Agent Output"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="Reflections",
        prompt=REACT_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Agent Output"

    # Test with custom prompt template string.
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["Agent Output"]),
        question="",
        scratchpad="",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="Reflections",
        prompt="{question} {scratchpad} {examples} {reflections} {max_steps}",
    )
    assert isinstance(out, str)
    assert out == "Agent Output"

