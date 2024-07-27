"""Unit tests for LATS functional module."""

import pytest

from agential.cog.lats.functional import (
    
    get_node_trajectory,
    get_unique_trajectories,
)


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

def test_get_node_trajectory(sample_nodes):
    root, child1, _ = sample_nodes
    expected_trajectory = "Root thought\nThought 1: Child1 thought\nAction 1: Lookup[topic]"
    assert get_node_trajectory(child1) == expected_trajectory

def test_get_unique_trajectories():
    failed_trajectories = [
        {"trajectory": "Path1", "final_answer": "Answer1"},
        {"trajectory": "Path2", "final_answer": "Answer1"},  # Duplicate answer.
        {"trajectory": "Path3", "final_answer": "Answer2"},
    ]
    assert get_unique_trajectories(failed_trajectories) == ["Path1", "Path3"]