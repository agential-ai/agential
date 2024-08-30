"""Unit tests for the Self-Refine general strategy."""

import pytest

from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.cog.self_refine.prompts import SELF_REFINE_INSTRUCTION_HOTPOTQA
from agential.cog.self_refine.strategies.general import SelfRefineGeneralStrategy
from agential.llm.llm import MockLLM


def test_init() -> None:
    """Tests init."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineGeneralStrategy(llm=llm)
    assert strategy.llm == llm
    assert strategy.patience == 1
    assert strategy.testing == False


def test_generate_answer() -> None:
    """Tests generate_answer."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineGeneralStrategy(llm=llm)

    question = "chicken"
    examples = "noodle"
    prompt = "soup"
    additional_keys = {}

    with pytest.raises(NotImplementedError):
        strategy.generate_answer(question, examples, prompt, additional_keys)


def test_generate_critique() -> None:
    """Tests generate_critique."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineGeneralStrategy(llm=llm)

    question = ""
    examples = ""
    answer = ""
    prompt = ""
    additional_keys = {}

    with pytest.raises(NotImplementedError):
        strategy.generate_critique(question, examples, answer, prompt, additional_keys)


def test_update_answer_based_on_critique() -> None:
    """Tests update_answer_based_on_critique."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineGeneralStrategy(llm=llm)

    question = "question"
    examples = "examples"
    answer = "answer"
    critique = "critique"
    prompt = "prompt"
    additional_keys = {""}

    with pytest.raises(NotImplementedError):
        strategy.update_answer_based_on_critique(
            question, examples, answer, critique, prompt, additional_keys
        )


def test_halting_condition() -> None:
    """Tests halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        strategy.halting_condition(True)


def test_reset() -> None:
    """Tests reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        strategy.reset()
