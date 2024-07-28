"""Unit tests for LATS functional module."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.lats.node import Node
from agential.cog.lats.functional import (
    _build_agent_prompt,
    _build_reflection_prompt,
    _build_value_prompt,
    _prompt_agent,
    _prompt_reflection,
    _prompt_value,
    get_node_trajectory,
    get_unique_trajectories,
)
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    LATS_INSTRUCTION_HOTPOTQA,
)


def test__build_reflection_prompt() -> None:
    """Tests the _build_reflection_prompt() function."""
    prompt = _build_reflection_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  # Use LATS specific examples
        prompt=LATS_INSTRUCTION_HOTPOTQA,  # Use LATS instruction
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_reflection() -> None:
    """Tests the _prompt_reflection() function."""
    out = _prompt_reflection(
        llm=FakeListChatModel(responses=["Reflection Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,  # Use LATS specific examples
        prompt=LATS_INSTRUCTION_HOTPOTQA,  # Use LATS instruction
    )
    assert isinstance(out, str)
    assert out == "Reflection Output"


def test__build_reflection_prompt() -> None:
    """Tests the _build_reflection_prompt() function."""
    prompt = _build_reflection_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_reflection() -> None:
    """Tests the _prompt_reflection() function."""
    out = _prompt_reflection(
        llm=FakeListChatModel(responses=["Reflection Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Reflection Output"


def test__build_value_prompt() -> None:
    """Tests the _build_value_prompt() function."""
    prompt = _build_value_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        failed_trajectories="Failed Trajectories",
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_value() -> None:
    """Tests the _prompt_value() function."""
    out = _prompt_value(
        llm=FakeListChatModel(responses=["Value Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        failed_trajectories="Failed Trajectories",
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Value Output"


def test__build_agent_prompt() -> None:
    """Tests the _build_agent_prompt() function."""
    prompt = _build_agent_prompt(
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_HOTPOTQA,
        reflections="Reflections",
    )
    assert isinstance(prompt, str)
    assert "Colorado orogeny" in prompt
    assert "elevation range" in prompt


def test__prompt_agent() -> None:
    """Tests the _prompt_agent() function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["Agent Output"]),
        question="What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
        trajectory="Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        reflections="Reflections",
        prompt=LATS_INSTRUCTION_HOTPOTQA,
    )
    assert isinstance(out, str)
    assert out == "Agent Output"


def test_get_node_trajectory() -> None:
    """Tests the get_node_trajectory() function."""
    root = Node(state={"thought": "Root thought", "action_type": "", "query": "", "observation": ""})
    child1 = Node(
        state={
            "thought": "Child1 thought",
            "action_type": "Lookup",
            "query": "topic",
            "observation": "",
        },
        parent=root,
    )
    child2 = Node(
        state={
            "thought": "Child2 thought",
            "action_type": "Finish",
            "query": "answer",
            "observation": "Answer correct",
        },
        parent=child1,
    )

    expected_trajectory = '\nThought 1: Child1 thought\nAction 1: Lookup[topic]\nThought 2: Child2 thought\nAction 2: Finish[answer]\nObservation 2: Answer correct'
    assert get_node_trajectory(child2) == expected_trajectory


def test_get_unique_trajectories() -> None:
    """Tests the get_unique_trajectories() function."""
    failed_trajectories = [
        {"trajectory": "Path1", "final_answer": "Answer1"},
        {"trajectory": "Path2", "final_answer": "Answer1"},  # Duplicate answer
        {"trajectory": "Path3", "final_answer": "Answer2"},
        {"trajectory": "Path4", "final_answer": "Answer3"},
        {"trajectory": "Path5", "final_answer": "Answer2"},  # Duplicate answer
        {"trajectory": "Path6", "final_answer": "Answer4"},
    ]

    # Test with max_unique=5
    result = get_unique_trajectories(failed_trajectories, max_unique=5)
    assert result == ["Path1", "Path3", "Path4", "Path6"]

    # Test with max_unique=2
    result = get_unique_trajectories(failed_trajectories, max_unique=2)
    assert result == ["Path1", "Path3"]

    # Test with empty list
    result = get_unique_trajectories([], max_unique=5)
    assert result == []

    # Test with all unique answers
    unique_trajectories = [
        {"trajectory": f"Path{i}", "final_answer": f"Answer{i}"} for i in range(1, 7)
    ]
    result = get_unique_trajectories(unique_trajectories, max_unique=5)
    assert result == [f"Path{i}" for i in range(1, 6)]  # Should return 5 trajectories
