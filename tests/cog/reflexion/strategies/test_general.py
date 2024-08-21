"""Unit tests for Reflexion general strategies."""

import pytest

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.reflexion.reflect import ReflexionCoTReflector
from agential.cog.reflexion.strategies.general import ReflexionCoTGeneralStrategy, ReflexionReActGeneralStrategy
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.metrics import PromptMetrics


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTGeneralStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3


def test_reflexion_cot_generate_thought() -> None:
    """Tests ReflexionCoTGeneralStrategy generate_thought."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = '\nThought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    gt_out = 'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    responses = [
        'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTGeneralStrategy(llm=llm)
    scratchpad, out, thought_metrics = strategy.generate_thought(
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out
    assert scratchpad == gt_scratchpad
    assert thought_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTGeneralStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        _, _, _ = strategy.generate_action(
            idx=0,
            scratchpad="",
            question=question,
            examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
            reflections="",
            prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
            additional_keys={},
        )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTGeneralStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        _, _, _ = strategy.generate_observation(
            idx=0, scratchpad="", action_type="", query="", key=""
        )


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTGeneralStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        strategy.halting_condition(idx=0, key="", answer="")


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTGeneralStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        strategy.reflect_condition(idx=0, reflect_strategy=None, key="", answer="")


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTGeneralStrategy reflect."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm, max_trials=3)

    gt_reflection_str = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\n\n(END PREVIOUS TRIAL)\n"
    reflections, reflection_str, reflection_metrics = strategy.reflect(
        scratchpad="",
        reflect_strategy="last_attempt",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert reflections == [""]
    assert reflection_str == gt_reflection_str
    assert reflection_metrics is None

    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionCoTGeneralStrategy(llm=llm, max_trials=3)

    gt_reflection_str = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    reflections, reflection_str, reflection_metrics = strategy.reflect(
        scratchpad="",
        reflect_strategy="reflexion",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert reflections == ["1"]
    assert reflection_str == gt_reflection_str
    assert reflection_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTGeneralStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTGeneralStrategy(llm=llm)

    strategy.reflector.reflections = ["1"]
    strategy.reset()

    assert strategy.reflector.reflections == []


def test_reflexion_react_init() -> None:
    """Tests ReflexionReactGeneralStrategy init."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

def test_reflexion_react_generate_thought() -> None:
    """Tests ReflexionReactGeneralStrategy generate_thought."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReactGeneralStrategy generate_action."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReactGeneralStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReactGeneralStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReactGeneralStrategy react_halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReactGeneralStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReactGeneralStrategy reflect."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

def test_reflexion_react_reset() -> None:
    """Tests ReflexionReactGeneralStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActGeneralStrategy(llm=llm)

