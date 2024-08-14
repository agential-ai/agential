"""Unit tests for LATS node module."""

import numpy as np
import pytest

from agential.cog.lats.node import Node
from agential.cog.lats.output import LATSReActStepOutput


def test_node_init() -> None:
    """Test node init."""
    node = Node()

    assert node.state == LATSReActStepOutput(
        thought="",
        action_type="",
        query="",
        observation="",
        answer="",
        external_tool_info={},
    )
    assert node.parent is None
    assert node.children == []
    assert node.visits == 0
    assert node.value == 0
    assert node.depth == 0
    assert not node.is_terminal
    assert node.reward == 0


def test_node_with_parent() -> None:
    """Test node init with parent."""
    parent = Node()
    child = Node(parent=parent)
    assert child.parent == parent
    assert child.depth == 1


def test_node_uct() -> None:
    """Test node uct."""
    parent = Node()
    parent.visits = 10
    child = Node(parent=parent)
    child.visits = 5
    child.value = 10
    expected_uct = 2 + np.sqrt(2 * np.log(10) / 5)
    assert child.uct() == pytest.approx(expected_uct)


def test_node_add_children() -> None:
    """Test node add_children."""
    parent = Node()
    child1 = Node()
    child2 = Node()
    parent.add_children([child1, child2])
    assert len(parent.children) == 2
    assert parent.children[0] == child1
    assert parent.children[1] == child2


def test_node_to_dict() -> None:
    """Test node to_dict."""
    gt_state = {
        "state": LATSReActStepOutput(
            thought="Test thought",
            action_type="Test action",
            query="Test query",
            observation="",
            answer="",
            external_tool_info={},
        ),
        "visits": 5,
        "value": 10,
        "depth": 2,
        "is_terminal": True,
        "reward": 1,
    }
    node = Node(
        state=LATSReActStepOutput(
            **{
                "thought": "Test thought",
                "action_type": "Test action",
                "query": "Test query",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        visits=5,
        value=10,
        depth=2,
        is_terminal=True,
        reward=1,
    )
    node_dict = node.to_dict()
    assert node_dict == gt_state
    assert node_dict["visits"] == 5
    assert node_dict["value"] == 10
    assert node_dict["depth"] == 2
    assert node_dict["is_terminal"] == True
    assert node_dict["reward"] == 1


def test_node_uct_zero_visits() -> None:
    """Test node uct with zero visits."""
    node = Node(value=5)
    assert node.uct() == 5


def test_node_depth_inheritance() -> None:
    """Test node depth inheritance."""
    root = Node()
    level1 = Node(parent=root)
    level2 = Node(parent=level1)
    level3 = Node(parent=level2)
    assert root.depth == 0
    assert level1.depth == 1
    assert level2.depth == 2
    assert level3.depth == 3
