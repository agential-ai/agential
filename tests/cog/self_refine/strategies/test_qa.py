"""Unit tests for Self-Refine QA strategies."""

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
from agential.llm.llm import MockLLM


def test_init() -> None:
    """Test SelfRefineQAStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy._prev_code_answer == ""
    assert strategy.patience_counter == 0
    assert not strategy._halt
    assert strategy._prompt_metrics == {
        "answer": None,
        "critique": None,
        "updated_answer": None,
    }


def test_generate() -> None:
    """Tests SelfRefineQAStrategy generate."""
    llm = MockLLM("gpt-3.5-turbo", responses=["Badr Hari"])
    strategy = SelfRefineQAStrategy(llm=llm)
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    answer = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt=SELF_REFINE_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert answer == "Badr Hari"
    assert strategy._prompt_metrics == {
        "answer": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
        "critique": None,
        "updated_answer": None,
    }


def test_generate_critique() -> None:
    """Tests SelfRefineQAStrategy generate_critique."""
    gt_critique = "1"
    responses = ["1"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
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
    assert strategy._prompt_metrics == {
        "answer": None,
        "critique": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
        "updated_answer": None,
    }

    # Test early stopping.
    gt_critique = "1"
    answer = "Mike Tyson"
    responses = ["1"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
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
    assert strategy._prompt_metrics == {
        "answer": None,
        "critique": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
        "updated_answer": None,
    }


def test_create_output_dict() -> None:
    """Tests SelfRefineQAStrategy create_output_dict."""
    strategy = SelfRefineQAStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    answer = "result = 42"
    critique = "Critique: Your solution is incorrect."
    output_dict = strategy.create_output_dict(answer, critique)
    assert output_dict == {
        "answer": answer,
        "critique": critique,
        "prompt_metrics": {"answer": None, "critique": None, "updated_answer": None},
    }


def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineQAStrategy update_answer_based_on_critique."""
    responses = ["1"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
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
    assert strategy._prompt_metrics == {
        "answer": None,
        "critique": None,
        "updated_answer": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        },
    }


def test_halting_condition() -> None:
    """Tests SelfRefineQAStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition() is False

    # Simulate the halting condition being met.
    strategy._halt = True
    assert strategy.halting_condition() is True


def test_reset() -> None:
    """Tests SelfRefineQAStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineQAStrategy(llm=llm, patience=2)

    strategy._prev_code_answer = "result = 42"
    strategy.patience_counter = 1
    strategy._halt = True
    strategy.reset()
    assert strategy._prev_code_answer == ""
    assert strategy.patience_counter == 0
    assert not strategy._halt
    assert strategy._prompt_metrics == {
        "answer": None,
        "critique": None,
        "updated_answer": None,
    }


def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    assert isinstance(SelfRefineHotQAStrategy(llm=llm), SelfRefineHotQAStrategy)
    assert isinstance(SelfRefineTriviaQAStrategy(llm=llm), SelfRefineTriviaQAStrategy)
    assert isinstance(SelfRefineAmbigNQStrategy(llm=llm), SelfRefineAmbigNQStrategy)
    assert isinstance(SelfRefineFEVERStrategy(llm=llm), SelfRefineFEVERStrategy)
