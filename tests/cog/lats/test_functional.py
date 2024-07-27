"""Unit tests for LATS functional module."""

import pytest

from agential.cog.lats.functional import (
    Node,
    upward_traversal,
    generate_prompt,
    get_node_trajectory,
    select_node,
    get_unique_trajectories,
    get_samples, 
    generate_new_states,
    expand_node,
    get_value,
    get_values,
    evaluate_node,
    rollout,
    backpropagate,
    preorder_traversal
)

# Mocking LLM for Test Scenarios (using langchain-community's FakeListChatModel)
from langchain_community.chat_models.fake import FakeListChatModel

class MockTask:
    def __init__(self):
        self.value_cache = {}  # Simulate value cache

    def value_prompt_wrap(self, x, y, z, a):
        return f"Value Prompt: {x}, {y}"  # Simplified for testing

    def value_outputs_unwrap(self, outputs):
        return float(outputs[0]) if outputs else 0  # Simulate value calculation

# Fixtures for Test Data
@pytest.fixture
def sample_node():
    return Node(state={"thought": "Initial thought", "action": "Search[term]", "observation": "Result found"})

@pytest.fixture
def sample_nodes():
    root = Node(state={"thought": "Root thought", "action": "", "observation": ""})
    child1 = Node(state={"thought": "Child1 thought", "action": "Lookup[topic]", "observation": ""}, parent=root)
    child2 = Node(state={"thought": "Child2 thought", "action": "Finish[answer]", "observation": "Answer correct"}, parent=root, is_terminal=True, reward=1)
    root.children = [child1, child2]
    return root, child1, child2

# Unit Tests
def test_upward_traversal(sample_nodes):
    root, child1, _ = sample_nodes
    assert upward_traversal(child1) == [root, child1]

def test_generate_prompt(sample_nodes):
    _, child1, _ = sample_nodes
    expected_prompt = "Thought 1: Child1 thought\nAction 1: Lookup[topic]"
    assert generate_prompt(upward_traversal(child1)) == expected_prompt

def test_get_node_trajectory(sample_nodes):
    root, child1, _ = sample_nodes
    expected_trajectory = "Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]"
    assert get_node_trajectory(child1) == expected_trajectory

def test_select_node(sample_nodes):
    root, _, child2 = sample_nodes  # Child2 is the terminal node with reward 1
    assert select_node(root) == child2



# Example using FakeListChatModel for mocking LLM responses
def test_get_value():
    task = MockTask()
    fake_llm = FakeListChatModel(responses=["0.8"])  # Simulate LLM response
    # Test without cache (initial call)
    assert get_value(task, "question", "answer", 1, cache_value=False) == 0.8
    # Test with cache (subsequent call)
    assert get_value(task, "question", "answer", 1, cache_value=True) == 0.8

def test_get_unique_trajectories():
    failed_trajectories = [
        {"trajectory": "Path1", "final_answer": "Answer1"},
        {"trajectory": "Path2", "final_answer": "Answer1"},  # Duplicate answer
        {"trajectory": "Path3", "final_answer": "Answer2"},
    ]
    assert get_unique_trajectories(failed_trajectories) == ["Path1", "Path3"]

def test_get_samples():
    # Setup: Fake LLM, MockTask, and global variables
    fake_llm = FakeListChatModel(responses=["Next action suggestion"])
    task = MockTask()
    global failed_trajectories, reflection_map
    failed_trajectories = [{"trajectory": "Some trajectory", "final_answer": "Wrong answer"}]
    reflection_map = []  # Empty for simplicity

    # Test standard prompt
    samples = get_samples(
        "What is the capital of France?",
        "Previous trajectory",
        "Current thought",
        {},  # additional_keys
        1,  # n_generate_sample
        "standard",
        "Observation",
    )
    assert samples == ["Current thoughtNext action suggestion"]

    # Test CoT prompt (without reflections)
    samples = get_samples(
        "What is the capital of France?",
        "Previous trajectory",
        "Current thought",
        {},
        1,
        "cot",
        "Observation",
    )
    # The CoT prompt should include "Chain of Thought:" and the additional keys
    assert "Chain of Thought:" in samples[0] 
    assert all(key in samples[0] for key in {})  # Check additional keys

    # Test CoT prompt (with reflections - this would need more mocking)
    # ... (This requires complex mocking of reflection generation and prompt building)

def test_generate_new_states(sample_node, monkeypatch):
    # Mock the `env.step` function and gpt function
    def mock_env_step(action):
        return f"Observation for {action}", 0, False, {}

    def mock_gpt(prompt, n, stop):
        return [f"Thought: New thought {i}\nAction: Action{i}" for i in range(n)]
    
    monkeypatch.setattr("agential.cog.lats.functional.gpt", mock_gpt)
    monkeypatch.setattr("agential.cog.lats.functional.env.step", mock_env_step)

    # Prepare for the test
    sample_node.question = "Sample Question"
    prompt_sample = "standard"
    n = 2  # Number of new states to generate

    # Run the function
    new_states = generate_new_states(sample_node, prompt_sample, n)

    # Assertions
    assert len(new_states) == n
    # Additional assertions can be added to check the contents of new states based on the mocked responses



# Test Expand Node
def test_expand_node(sample_node, monkeypatch):
    # Mock the generate_new_states function and node properties
    def mock_generate_new_states(node, prompt_sample, n_generate_sample):
        return [
            Node(
                state={"thought": "New thought 1", "action": "Action1"},
                question=node.question,
                parent=node,
                depth=node.depth + 1,
            ),
            Node(
                state={"thought": "New thought 2", "action": "Action2"},
                question=node.question,
                parent=node,
                depth=node.depth + 1,
            ),
        ]

    monkeypatch.setattr(
        "agential.cog.lats.functional.generate_new_states", mock_generate_new_states
    )

    # Prepare for the test
    sample_node.question = "Sample Question"
    prompt_sample = "standard"
    n_generate_sample = 2

    # Run the function
    children_nodes = expand_node(sample_node, prompt_sample, n_generate_sample)

    # Assertions
    assert len(children_nodes) == 2
    assert all(isinstance(child, Node) for child in children_nodes)
    assert all(child.parent == sample_node for child in children_nodes)

# Test Get Values
def test_get_values(sample_node):
    task = MockTask()
    fake_llm = FakeListChatModel(responses=["0.8", "0.6"])  # Simulate LLM responses
    child_prompts = [
        sample_node.question + generate_prompt(upward_traversal(child))
        for child in sample_node.children
    ]
    assert get_values(task, sample_node.question, child_prompts, 1) == [0.8, 0.6]


# Test Evaluate Node
def test_evaluate_node(sample_node, monkeypatch):
    # Mock the get_values function
    def mock_get_values(task, x, ys, n_evaluate_sample):
        return [0.8, 0.6]

    monkeypatch.setattr("agential.cog.lats.functional.get_values", mock_get_values)

    task = MockTask()
    n_evaluate_sample = 1
    # Add children to the sample node for evaluation
    sample_node.children = [Node(), Node()]  

    assert evaluate_node(sample_node, task, n_evaluate_sample) == 0.7


# Test Rollout
def test_rollout(sample_node, monkeypatch):
    task = MockTask()
    prompt_sample = "standard"

    # Mock the generate_new_states, get_values functions, and node properties
    def mock_generate_new_states(node, prompt_sample, n):
        return [Node(is_terminal=True, reward=1)]

    def mock_get_values(task, x, ys, n_evaluate_sample):
        return [0.9]

    monkeypatch.setattr(
        "agential.cog.lats.functional.generate_new_states", mock_generate_new_states
    )
    monkeypatch.setattr("agential.cog.lats.functional.get_values", mock_get_values)

    reward, final_node = rollout(sample_node, task, 1, prompt_sample)

    assert reward == 1
    assert final_node.is_terminal

# Test Backpropagate
def test_backpropagate(sample_nodes):
    root, child1, child2 = sample_nodes
    value = 0.8
    backpropagate(child2, value)
    assert child2.value == 0.8
    assert child2.visits == 1
    assert root.value == 0.8
    assert root.visits == 1

# Test Preorder Traversal
def test_preorder_traversal(sample_nodes):
    root, child1, child2 = sample_nodes
    nodes = preorder_traversal(root)
    assert nodes == [root, child1, child2]