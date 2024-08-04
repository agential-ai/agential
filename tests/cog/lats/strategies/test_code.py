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


def test_parse_latest_implement() -> None:
    """Test parse_latest_implement function."""
    
    # Test case with single implementation.
    single_impl = """
    Some text
    Implement[```python
    def add(a, b):
        return a + b
    ```]
    More text
    """
    assert parse_latest_implement(single_impl) == "def add(a, b):\n        return a + b"

    # Test case with multiple implementations.
    multiple_impl = """
    Implement[```python
    def subtract(a, b):
        return a - b
    ```]
    Some text
    Implement[```python
    def multiply(a, b):
        return a * b
    ```]
    """
    assert parse_latest_implement(multiple_impl) == "def multiply(a, b):\n        return a * b"

    # Test case with no implementation.
    no_impl = "Some text without any implementation"
    assert parse_latest_implement(no_impl) == ""

    # Test case with empty implementation.
    empty_impl = "Implement[```python\n```]"
    assert parse_latest_implement(empty_impl) == ""

    # Test case with multiple lines in implementation.
    multi_line_impl = """
    Implement[```python
    def complex_function(x):
        if x > 0:
            return x * 2
        else:
            return x * -1
    ```]
    """
    expected_multi_line = """def complex_function(x):
        if x > 0:
            return x * 2
        else:
            return x * -1"""
    assert parse_latest_implement(multi_line_impl) == expected_multi_line


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


def test_parse_code_action() -> None:
    """Test parse_code_action function."""
    test_cases = [
        {
            "input": "Implement[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Implement", "def add(a, b): return a + b"),
        },
        {
            "input": "TEST[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Test", "assert add(2, 3) == 5"),
        },
        {
            "input": "finish[```python\nprint('Done')\n```]",
            "expected": ("Finish", "print('Done')"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Implement[```python\n \n```]",
            "expected": ("Implement", ""),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
    ]

    for case in test_cases:
        result = parse_code_action(case["input"])
        assert result == case["expected"]

    exception_case = "Implement[```python\nincomplete code"
    result = parse_code_action(exception_case)
    assert result == ('Implement', 'incomplete code')


def test_parse_code_value():
    """Test the parse_code_value function."""
    # Test valid value strings.
    valid_input = (
        "Some text. Explanation: This is the explanation. Correctness score: 5"
    )
    assert parse_code_value(valid_input) == ("This is the explanation.", 5)

    # Test invalid value strings.
    assert parse_code_value("No explanation or score") == ("Explanation not found", 0)
    assert parse_code_value("Explanation: Only explanation") == (
        "Explanation not found",
        0,
    )
    assert parse_code_value("Correctness score: 5") == ("Explanation not found", 0)

    # Test edge cases.
    assert parse_code_value("Explanation: Empty. Correctness score: 0") == ("Empty.", 0)
    assert parse_code_value(
        "Explanation: Multi-line\nexplanation. Correctness score: 10"
    ) == ("Multi-line\nexplanation.", 10)

    # Test with unexpected format.
    assert parse_code_value("Explanation: Tricky: score. Correctness score: 7") == (
        "Tricky: score.",
        7,
    )