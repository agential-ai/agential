"""Unit tests for Reflexion Code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.code import (
    ReflexionCoTCodeStrategy,
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActCodeStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
    parse_code_action_cot,
    parse_code_action_react,
)
from agential.cog.prompts.benchmark.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_REACT
)
from agential.cog.prompts.agent.reflexion import (
    REFLEXION_COT_INSTRUCTION_MBPP,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_REACT_INSTRUCTION_MBPP,
    REFLEXION_COT_REFLECT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP
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
    """Tests ReflexionCoTCodeStrategy init."""
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
    """Tests ReflexionCoTCodeStrategy generate."""
    question = "Write a python function to find the first repeated character in a given string."
    key = """assert first_repeated_char("abcabc") == "a"
    assert first_repeated_char("abc") == None
    assert first_repeated_char("123123") == "1\""""

    gt_out = "Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far to identify the first repeated character."
    gt_scratchpad = "\nThought: Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far to identify the first repeated character."
    responses = [
        "Let's think step by step. We need to iterate through the string and keep track of characters we have seen so far to identify the first repeated character.\nAction: Finish[\n```python\ndef first_repeated_char(s):\n    seen = set()\n    for char in s:\n        if char in seen:\n            return char\n        seen.add(char)\n    return None\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTCodeStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=MBPP_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_MBPP,
        additional_keys={"tests": key},
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""
    

def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTCodeStrategy generate_action."""


def test_reflexion_cot_generate_action_humaneval() -> None:
    """Tests ReflexionCoTHEvalStrategy generate_action."""


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTCodeStrategy generate_observation."""


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTCodeStrategy create_output_dict."""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTCodeStrategy halting_condition."""


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTCodeStrategy reset."""


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTCodeStrategy reflect."""


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTCodeStrategy reflect_condition."""


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTCodeStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    humaneval_strategy = ReflexionCoTHEvalStrategy(llm=llm)
    mbpp_strategy = ReflexionCoTMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReflexionCoTHEvalStrategy)
    assert isinstance(mbpp_strategy, ReflexionCoTMBPPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActCodeStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActCodeStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActCodeStrategy generate."""


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActCodeStrategy generate_action."""


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActCodeStrategy generate_observation."""


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActCodeStrategy create_output_dict."""


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActCodeStrategy react_create_output_dict."""


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActCodeStrategy halting_condition."""


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActCodeStrategy react_halting_condition."""


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActCodeStrategy reset."""


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActCodeStrategy reflect."""


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActCodeStrategy reflect_condition."""


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActCodeStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    humaneval_strategy = ReflexionReActHEvalStrategy(llm=llm)
    mbpp_strategy = ReflexionReActMBPPStrategy(llm=llm)

    assert isinstance(humaneval_strategy, ReflexionReActHEvalStrategy)
    assert isinstance(mbpp_strategy, ReflexionReActMBPPStrategy)
