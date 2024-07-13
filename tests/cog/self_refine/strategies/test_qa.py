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
    SelfRefineAmbigNQStrategy,
    SelfRefineFEVERStrategy,
    SelfRefineHotQAStrategy,
    SelfRefineQAStrategy,
    SelfRefineTriviaQAStrategy,
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
    llm = FakeListChatModel(responses=["Badr Hari"])
    strategy = SelfRefineQAStrategy(llm=llm)
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    answer = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt=SELF_REFINE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert answer == "Badr Hari"

def test_generate_critique() -> None:
    """Tests SelfRefineQAStrategy generate_critique."""
    gt_critique = "1"
    responses = [
        "1"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm)
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'
    answer = "Mike Tyson"

    critique = strategy.generate_critique(
        question=question,
        examples=HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert critique == gt_critique
    assert not strategy._halt
    assert strategy._prev_code_answer == answer
    assert strategy.patience_counter == 0

    # Test early stopping.
    gt_critique = "1"
    answer = "Mike Tyson"
    responses = [
        "1"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm, patience=1)
    strategy._prev_code_answer = "Mike Tyson"
    critique = strategy.generate_critique(
        question=question,
        examples=HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert critique == gt_critique
    assert strategy.patience_counter == 1
    assert strategy._halt is True
    assert strategy._prev_code_answer == "Mike Tyson"


def test_create_output_dict() -> None:
    """Tests SelfRefineQAStrategy create_output_dict."""
    strategy = SelfRefineQAStrategy(llm=FakeListChatModel(responses=[]))
    answer = "result = 42"
    critique = "Critique: Your solution is incorrect."
    output_dict = strategy.create_output_dict(answer, critique)
    assert output_dict == {"answer": answer, "critique": critique}


def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineQAStrategy update_answer_based_on_critique."""
    responses = ["1"]
    llm = FakeListChatModel(responses=responses)
    strategy = SelfRefineQAStrategy(llm=llm)
    question = "Sample question"
    answer = "Mike Tyson"
    critique = "Critique: Your solution is incorrect."

    new_answer = strategy.update_answer_based_on_critique(
        question=question,
        examples=HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
        answer=answer,
        critique=critique,
        prompt=SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert new_answer == "1"

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
