"""Unit tests for Self-Refine QA strategies."""
from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.cog.self_refine.prompts import (
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.self_refine.strategies.qa import (
    SelfRefineHotQAStrategy,
    SelfRefineTriviaQAStrategy,
    SelfRefineAmbigNQStrategy,
    SelfRefineFEVERStrategy,
    SelfRefineQAStrategy,
)


def test_init() -> None:
    """Test SelfRefineQAStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy._prev_code_answer == ""
    assert strategy.patience_counter == 0
    assert not strategy._halt

def test_generate() -> None:
    """Tests SelfRefineQAStrategy generate."""

def test_generate_critique() -> None:
    """Tests SelfRefineQAStrategy generate_critique."""

def test_create_output_dict() -> None:
    """Tests SelfRefineQAStrategy create_output_dict."""
    strategy = SelfRefineQAStrategy(llm=FakeListChatModel(responses=[]))
    answer = "result = 42"
    critique = "Critique: Your solution is incorrect."
    output_dict = strategy.create_output_dict(answer, critique)
    assert output_dict == {"answer": answer, "critique": critique}

def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineQAStrategy update_answer_based_on_critique."""

def test_halting_condition() -> None:
    """Tests SelfRefineQAStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition() is False

    # Simulate the halting condition being met.
    strategy._halt = True
    assert strategy.halting_condition() is True

def test_reset() -> None:
    """Tests SelfRefineQAStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=2)

    strategy._prev_code_answer = "result = 42"
    strategy.patience_counter = 1
    strategy._halt = True
    strategy.reset()
    assert strategy._prev_code_answer == ""
    assert strategy.patience_counter == 0
    assert not strategy._halt

def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = FakeListChatModel(responses=[])
    assert isinstance(SelfRefineHotQAStrategy(llm=llm), SelfRefineHotQAStrategy)
    assert isinstance(SelfRefineTriviaQAStrategy(llm=llm), SelfRefineTriviaQAStrategy)
    assert isinstance(SelfRefineAmbigNQStrategy(llm=llm), SelfRefineAmbigNQStrategy)
    assert isinstance(SelfRefineFEVERStrategy(llm=llm), SelfRefineFEVERStrategy)