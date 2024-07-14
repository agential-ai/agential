"""Unit tests for Self-Refine code strategies."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.fewshots.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_POT
from agential.cog.self_refine.prompts import (
    HUMANEVAL_CRITIQUE_FEWSHOT_EXAMPLES,
    HUMANEVAL_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_INSTRUCTION_HUMANEVAL,
    SELF_REFINE_REFINE_INSTRUCTION_HUMANEVAL,
)
from agential.cog.self_refine.strategies.code import (
    SelfRefineCodeStrategy,
    SelfRefineHEvalStrategy,
    SelfRefineMBPPStrategy,
)   


def test_init() -> None:
    """Test SelfRefineCodeStrategy initialization."""

def test_generate() -> None:
    """Tests SelfRefineCodeStrategy generate."""

def test_generate_critique() -> None:
    """Tests SelfRefineCodeStrategy generate_critique."""

def test_create_output_dict() -> None:
    """Tests SelfRefineCodeStrategy create_output_dict."""

def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineCodeStrategy update_answer_based_on_critique."""
    gt_answer = 'from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False'
    responses = [
        '```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n```'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = SelfRefineCodeStrategy(llm=llm, patience=2)
    new_answer = strategy.update_answer_based_on_critique(
        question="",
        examples="",
        answer="",
        critique="",
        prompt="",
        additional_keys={}
    )
    assert new_answer == gt_answer

def test_halting_condition() -> None:
    """Tests SelfRefineCodeStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineCodeStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition() is False

    # Simulate the halting condition being met.
    strategy._halt = True
    assert strategy.halting_condition() is True

def test_reset() -> None:
    """Tests SelfRefineCodeStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineCodeStrategy(llm=llm, patience=2)

    strategy._prev_code_answer = "result = 42"
    strategy.patience_counter = 1
    strategy._halt = True
    strategy.reset()
    assert strategy._prev_code_answer == ""
    assert strategy.patience_counter == 0
    assert not strategy._halt

def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = FakeListChatModel(responses=[])
    assert isinstance(SelfRefineHEvalStrategy(llm=llm), SelfRefineHEvalStrategy)
    assert isinstance(SelfRefineMBPPStrategy(llm=llm), SelfRefineMBPPStrategy)
