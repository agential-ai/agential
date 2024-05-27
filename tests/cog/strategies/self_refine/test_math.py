"""Unit tests for Self-Refine math strategies."""
from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.strategies.self_refine.math import (
    SelfRefineGSM8KStrategy,
    SelfRefineMathStrategy
)
from agential.cog.prompts.benchmarks.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT
from agential.cog.prompts.self_refine import SELF_REFINE_INSTRUCTION_GSM8K


def test_init() -> None:
    """Test SelfRefineMathStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineMathStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy._prev_code_answer is None
    assert strategy.patience_counter == 0
    assert not strategy._halt


def test_generate() -> None:
    """Tests SelfRefineMathStrategy generate."""
    llm = FakeListChatModel(responses=["```python\nresult = 42\n```"])
    strategy = SelfRefineMathStrategy(llm=llm)
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"

    answer = strategy.generate(
        question=question, 
        examples=GSM8K_FEWSHOT_EXAMPLES_POT, 
        prompt=SELF_REFINE_INSTRUCTION_GSM8K, 
        additional_keys={}
    )
    assert answer == "result = 42"


def test_generate_critique() -> None:
    """Tests SelfRefineMathStrategy generate_critique."""

def test_create_output_dict() -> None:
    """Tests SelfRefineMathStrategy create_output_dict."""

def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineMathStrategy update_answer_based_on_critique."""

def test_halting_condition() -> None:
    """Tests SelfRefineMathStrategy halting_condition."""

def test_reset() -> None:
    """Tests SelfRefineMathStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""