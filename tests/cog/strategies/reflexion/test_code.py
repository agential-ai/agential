"""Unit tests for Reflexion Code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.code import (
    parse_code_action_cot,
    parse_code_action_react,
    ReflexionCoTCodeStrategy,
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActCodeStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
)


def test_parse_code_action_cot() -> None:
    """Tests parse_code_action_cot."""
    # Test case 1: Correct Finish action.
    action = "Finish```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("Finish", "print('Hello, World!')")

    # Test case 2: No action type.
    action = "```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("", "")

    # Test case 3: Incorrect action type.
    action = "End```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("", "")

    # Test case 4: Finish action with mixed case.
    action = "fIniSh```python\nprint('Hello, World!')\n```"
    assert parse_code_action_cot(action) == ("Finish", "print('Hello, World!')")


def test_parse_code_action_react() -> None:
    """Tests parse_code_action_react."""
    # Test case 1: Correct Finish action.
    action = "Finish```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("Finish", "print('Hello, World!')")

    # Test case 2: Correct Implement action.
    action = "Implement```python\nx = 10\n```"
    assert parse_code_action_react(action) == ("Implement", "x = 10")

    # Test case 3: Correct Test action.
    action = "Test```python\nassert x == 10\n```"
    assert parse_code_action_react(action) == ("Test", "assert x == 10")

    # Test case 4: No action type.
    action = "```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("", "")

    # Test case 5: Incorrect action type.
    action = "End```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("", "")

    # Test case 6: Mixed case action types.
    action = "FiNiSh```python\nprint('Hello, World!')\n```"
    assert parse_code_action_react(action) == ("Finish", "print('Hello, World!')")

    action = "imPlEmEnT```python\nx = 10\n```"
    assert parse_code_action_react(action) == ("Implement", "x = 10")

    action = "tEsT```python\nassert x == 10\n```"
    assert parse_code_action_react(action) == ("Test", "assert x == 10")


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTMathStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTMathStrategy generate."""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTMathStrategy generate_action."""


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTMathStrategy generate_observation."""


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTMathStrategy create_output_dict."""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTMathStrategy halting_condition."""


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTMathStrategy reset."""


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTMathStrategy reflect."""


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTMathStrategy reflect_condition."""


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTMathStrategy instantiate strategies."""


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActMathStrategy init."""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActMathStrategy generate."""


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActMathStrategy generate_action."""


def test_reflexion_cot_generate_action_humaneval() -> None:
    """Tests ReflexionCoTHEvalStrategy generate_action."""


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActMathStrategy generate_observation."""


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActMathStrategy create_output_dict."""


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActMathStrategy react_create_output_dict."""


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy halting_condition."""


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy react_halting_condition."""


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActMathStrategy reset."""


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActMathStrategy reflect."""


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActMathStrategy reflect_condition."""


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActMathStrategy instantiate strategies."""
