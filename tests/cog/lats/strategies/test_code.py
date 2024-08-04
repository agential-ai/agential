"""Unit tests for LATS Code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from agential.cog.lats.strategies.code import (
    LATSCodeStrategy,
    LATSHEvalStrategy,
    LATSMBPPStrategy,
    parse_latest_implement,
    get_node_trajectory_code,
    parse_code_action,
    parse_code_value,
)
from agential.cog.react.output import ReActOutput
from agential.cog.lats.node import Node


def test_get_node_trajectory_code() -> None:
    """Tests the get_node_trajectory_code() function."""
    root = Node(
        state=ReActOutput(
            **{
                "thought": "Root thought",
                "action_type": "",
                "query": "",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        )
    )
    child1 = Node(
        state=ReActOutput(
            **{
                "thought": "Child1 thought",
                "action_type": "Lookup",
                "query": "topic",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        parent=root,
    )
    child2 = Node(
        state=ReActOutput(
            **{
                "thought": "Child2 thought",
                "action_type": "Finish",
                "query": "answer",
                "observation": "Answer correct",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        parent=child1,
    )

    expected_trajectory = "\nThought 1: Child1 thought\nAction 1: Lookup[\n```python\ntopic\n```\n]\nThought 2: Child2 thought\nAction 2: Finish[\n```python\nanswer\n```\n]\nObservation 2: Answer correct"
    assert get_node_trajectory_code(child2) == expected_trajectory

    # Test root node.
    root = Node()
    assert get_node_trajectory_code(root) == ""