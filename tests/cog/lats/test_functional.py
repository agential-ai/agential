"""Unit tests for LATS functional module."""

import pytest

from langchain_core.messages.human import HumanMessage
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


# Mock Data
MOCK_QUESTION = "What is the meaning of life?"
MOCK_EXAMPLES = "Here are some examples..."
MOCK_TRAJECTORY = "Thought: ... Action: ... Observation: ..."
MOCK_REFLECTIONS = "Reflection 1: ... Reflection 2: ..."
MOCK_FAILED_TRAJECTORIES = "Failed 1: ... Failed 2: ..."
MOCK_PROMPT_TEMPLATE = "Question: {question}\nExamples: {examples}"
MOCK_ADDITIONAL_KEYS = {"key1": "value1", "key2": "value2"}


# Fixtures
@pytest.fixture
def mock_chat_model():
    return FakeListChatModel(
        responses=["Reflection Output", "Value Output", "Agent Output"]
    )


@pytest.fixture
def mock_prompt_template():
    class MockPromptTemplate:
        def __init__(self, template):
            self.template = template

        def format(self, **kwargs):
            return self.template.format(**kwargs)  # Simply format the template

    return MockPromptTemplate(MOCK_PROMPT_TEMPLATE)


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


def test_build_reflection_prompt(mock_prompt_template):
    prompt = _build_reflection_prompt(
        MOCK_QUESTION,
        MOCK_EXAMPLES,
        MOCK_TRAJECTORY,
        mock_prompt_template,
        MOCK_ADDITIONAL_KEYS,
    )
    assert prompt == MOCK_PROMPT_TEMPLATE  # No mocking, so the raw template is returned


def test_prompt_reflection(mock_chat_model, mock_prompt_template):
    reflection = _prompt_reflection(
        mock_chat_model,
        MOCK_QUESTION,
        MOCK_EXAMPLES,
        MOCK_TRAJECTORY,
        mock_prompt_template,
        MOCK_ADDITIONAL_KEYS,
    )
    mock_chat_model.assert_called_once_with(
        [HumanMessage(content=MOCK_PROMPT_TEMPLATE)]
    )
    assert reflection == "Reflection Output"


def test_build_value_prompt(mock_prompt_template):
    prompt = _build_value_prompt(
        MOCK_QUESTION,
        MOCK_EXAMPLES,
        MOCK_TRAJECTORY,
        MOCK_FAILED_TRAJECTORIES,
        MOCK_PROMPT_TEMPLATE,
        MOCK_ADDITIONAL_KEYS,
    )
    mock_prompt_template.from_template.assert_called_once_with(MOCK_PROMPT_TEMPLATE)
    mock_prompt_template.from_template().format.assert_called_once_with(
        question=MOCK_QUESTION,
        examples=MOCK_EXAMPLES,
        trajectory=MOCK_TRAJECTORY,
        failed_trajectories=MOCK_FAILED_TRAJECTORIES,
        **MOCK_ADDITIONAL_KEYS,
    )
    assert prompt == MOCK_PROMPT_TEMPLATE


def test_build_agent_prompt(mock_prompt_template):
    prompt = _build_agent_prompt(
        MOCK_QUESTION,
        MOCK_EXAMPLES,
        MOCK_TRAJECTORY,
        MOCK_REFLECTIONS,
        MOCK_PROMPT_TEMPLATE,
        MOCK_ADDITIONAL_KEYS,
    )
    mock_prompt_template.from_template.assert_called_once_with(MOCK_PROMPT_TEMPLATE)
    mock_prompt_template.from_template().format.assert_called_once_with(
        question=MOCK_QUESTION,
        examples=MOCK_EXAMPLES,
        trajectory=MOCK_TRAJECTORY,
        reflections=MOCK_REFLECTIONS,
        **MOCK_ADDITIONAL_KEYS,
    )
    assert prompt == MOCK_PROMPT_TEMPLATE


def test_prompt_value(mock_llm, mock_prompt_template):
    mock_llm.return_value.content = "Value Output"
    value = _prompt_value(
        mock_llm,
        MOCK_QUESTION,
        MOCK_EXAMPLES,
        MOCK_TRAJECTORY,
        MOCK_FAILED_TRAJECTORIES,
        MOCK_PROMPT_TEMPLATE,
        MOCK_ADDITIONAL_KEYS,
    )
    mock_llm.assert_called_once_with([HumanMessage(content=MOCK_PROMPT_TEMPLATE)])
    assert value == "Value Output"


def test_prompt_agent(mock_llm, mock_prompt_template):
    mock_llm.return_value.content = "Agent Output"
    agent = _prompt_agent(
        mock_llm,
        MOCK_QUESTION,
        MOCK_EXAMPLES,
        MOCK_TRAJECTORY,
        MOCK_REFLECTIONS,
        MOCK_PROMPT_TEMPLATE,
        MOCK_ADDITIONAL_KEYS,
    )
    mock_llm.assert_called_once_with([HumanMessage(content=MOCK_PROMPT_TEMPLATE)])
    assert agent == "Agent Output"
