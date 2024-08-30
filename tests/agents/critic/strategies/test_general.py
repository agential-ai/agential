"""Unit tests for CRITIC general strategy."""

import pytest

from agential.agents.critic.strategies.general import CriticGeneralStrategy
from agential.llm.llm import BaseLLM, MockLLM


def test_init() -> None:
    """Test initialization of the CRITIC general strategy."""
    strategy = CriticGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate_answer() -> None:
    """Test generate_answer()."""
    strategy = CriticGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    with pytest.raises(NotImplementedError):
        strategy.generate_answer(
            question="What is the capital of France?",
            examples="Example 1: ...",
            prompt="Please answer the following question:",
            additional_keys={"key1": "value1"},
        )


def test_generate_critique() -> None:
    """Test generate_critique()."""
    strategy = CriticGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    with pytest.raises(NotImplementedError):
        strategy.generate_critique(
            idx=0,
            question="What is the capital of France?",
            examples="Example 1: ...",
            answer="Paris",
            critique="Previous critique",
            prompt="Please critique the following answer:",
            additional_keys={"key1": "value1"},
            use_tool=False,
            max_interactions=5,
        )


def test_update_answer_based_on_critique() -> None:
    """Test update_answer_based_on_critique()."""
    strategy = CriticGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    with pytest.raises(NotImplementedError):
        strategy.update_answer_based_on_critique(
            question="What is the capital of France?",
            examples="Example 1: ...",
            answer="Paris",
            critique="Previous critique",
            prompt="Please update the following answer based on the critique:",
            additional_keys={"key1": "value1"},
            external_tool_info={"tool_key": "tool_value"},
        )
