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
    SelfRefineHEvalStrategy,
    SelfRefineMBPPStrategy,
    SelfRefineCodeStrategy
)   


def test_init() -> None:
    """Test SelfRefinecodeStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineCodeStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy._prev_code_answer == ""
    assert strategy.patience_counter == 0
    assert not strategy._halt

def test_generate() -> None:
    """Tests SelfRefineCodeStrategy generate."""
    llm = FakeListChatModel(responses=['from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False\n'
])
    
    strategy = SelfRefineCodeStrategy(llm=llm)
    
    question = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
    
    answer = strategy.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=SELF_REFINE_INSTRUCTION_HUMANEVAL,
        additional_keys={},
    )
    assert answer == 'from typing import List\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n    for i in range(len(numbers) - 1):\n        if abs(numbers[i] - numbers[i + 1]) < threshold:\n            return True\n    return False'




def test_generate_critique() -> None:
    """Tests SelfRefineCodeStrategy generate_critique."""

def test_create_output_dict() -> None:
    """Tests SelfRefineCodeStrategy create_output_dict."""

def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineCodeStrategy update_answer_based_on_critique."""

def test_halting_condition() -> None:
    """Tests SelfRefineCodeStrategy halting_condition."""

def test_reset() -> None:
    """Tests SelfRefineCodeStrategy reset."""
    
def test_instantiate_strategies() -> None:
    """Test instantiate all Code strategies."""