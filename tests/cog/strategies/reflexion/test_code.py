"""Unit tests for Reflexion Code strategies."""


from agential.cog.strategies.reflexion.code import (
    parse_code_action_cot,
    parse_code_action_react,
)

def test_parse_code_action_cot() -> None:
    """Tests parse_code_action_cot."""
    # Test case 1: Correct Finish action
    action = "Finish```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("Finish", "print('Hello, World!')")

    # Test case 2: No action type
    action = "```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("", "")

    # Test case 3: Incorrect action type
    action = "End```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("", "")

    # Test case 4: Finish action with mixed case
    action = "fIniSh```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("Finish", "print('Hello, World!')")


def test_parse_code_action_react() -> None:
    """Tests parse_code_action_react."""
    # Test case 1: Correct Finish action
    action = "Finish```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("Finish", "print('Hello, World!')")

    # Test case 2: Correct Implement action
    action = "Implement```python\nx = 10\n```"
    assert parse_code_action_react(action) == ("Implement", "x = 10")

    # Test case 3: Correct Test action
    action = "Test```python\nassert x == 10\n```"
    assert parse_code_action_react(action) == ("Test", "assert x == 10")

    # Test case 4: No action type
    action = "```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("", "")

    # Test case 5: Incorrect action type
    action = "End```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("", "")

    # Test case 6: Mixed case action types
    action = "FiNiSh```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("Finish", "print('Hello, World!')")

    action = "imPlEmEnT```python\nx = 10\n```"
    assert parse_code_action_react(action) == ("Implement", "x = 10")

    action = "tEsT```python\nassert x == 10\n```"
    assert parse_code_action_react(action) == ("Test", "assert x == 10")
