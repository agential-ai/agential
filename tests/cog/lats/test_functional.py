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


from agential.cog.lats.prompts import LATS_INSTRUCTION_HOTPOTQA, HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT


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
    expected_trajectory = [
        "Root thought",
        "Thought 1: Child1 thought\nAction 1: Lookup[topic]"
    ]
    assert get_node_trajectory(child1) == expected_trajectory


def test_get_unique_trajectories():
    failed_trajectories = [
        {"trajectory": "Path1", "final_answer": "Answer1"},
        {"trajectory": "Path2", "final_answer": "Answer1"},  # Duplicate answer.
        {"trajectory": "Path3", "final_answer": "Answer2"},
    ]
    assert get_unique_trajectories(failed_trajectories) == ["Path1", "Path3"]



def test__build_reflection_prompt():
    prompt = _build_reflection_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  # Use LATS specific examples
        prompt=LATS_INSTRUCTION_HOTPOTQA,          # Use LATS instruction
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_reflection():
    out = _prompt_reflection(
        llm=FakeListChatModel(responses=["Reflection Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  # Use LATS specific examples
        prompt=LATS_INSTRUCTION_HOTPOTQA,          # Use LATS instruction
    )
    assert isinstance(out, str)
    assert out == "Reflection Output"

def test__build_reflection_prompt():
    prompt = _build_reflection_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples= HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT, # Update to LATS specific reflection examples
        prompt=LATS_INSTRUCTION_HOTPOTQA,          # Use LATS instruction
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_reflection():
    out = _prompt_reflection(
        llm=FakeListChatModel(responses=["Reflection Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples= HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT, # Update to LATS specific reflection examples
        prompt=LATS_INSTRUCTION_HOTPOTQA,          # Use LATS instruction
    )
    assert isinstance(out, str)
    assert out == "Reflection Output"



def test__build_value_prompt():
    prompt = _build_value_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples= HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        failed_trajectories="Failed Trajectories",
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt
    

def test__prompt_value():
    out = _prompt_value(
        llm=FakeListChatModel(responses=["Value Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples= HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  
        failed_trajectories="Failed Trajectories",
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Value Output"


def test__build_agent_prompt():
    prompt = _build_agent_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples= HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflections="Reflections",
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_agent():
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["Agent Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples= HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  
        reflections="Reflections",
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Agent Output"