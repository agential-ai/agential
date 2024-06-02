"""Unit tests for ReAct math strategies."""

from agential.cog.strategies.react.math import (
    parse_math_action,
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
)


def test_parse_math_action() -> None:
    """Test parse_math_action."""
    test_cases = [
        {
            "input": "Calculate[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Calculate", "def add(a, b): return a + b"),
        },
        {
            "input": "Finish[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Finish", "assert add(2, 3) == 5"),
        },
        {
            "input": "Finish[```python\nThe function is complete.\n```]",
            "expected": ("Finish", "The function is complete."),
        },
        {
            "input": "calculate[```python\ndef subtract(a, b): return a - b\n```]",
            "expected": ("Calculate", "def subtract(a, b): return a - b"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Calculate[```python\nassert subtract(5, 3) == 2\n```]",
            "expected": ("Calculate", "assert subtract(5, 3) == 2"),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
        {
            "input": "Finish[```python\n \n```]",
            "expected": ("Finish", ""),
        },
        {
            "input": "Calculate[```python\nfor i in range(10):\n    print(i)\n```]",
            "expected": ("Calculate", "for i in range(10):\n    print(i)"),
        },
    ]

    for case in test_cases:
        result = parse_math_action(case["input"])
        assert result == case["expected"]
