"""Unit tests for CoT QA strategies."""

from agential.llm.llm import MockLLM
from agential.prompting.cot.strategies.qa import (
    CoTAmbigNQStrategy,
    CoTFEVERStrategy,
    CoTHotQAStrategy,
    CoTTriviaQAStrategy,
)


def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    hotqa_strategy = CoTHotQAStrategy(llm=llm)
    triviaqa_strategy = CoTTriviaQAStrategy(llm=llm)
    ambignq_strategy = CoTAmbigNQStrategy(llm=llm)
    fever_strategy = CoTFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, CoTHotQAStrategy)
    assert isinstance(triviaqa_strategy, CoTTriviaQAStrategy)
    assert isinstance(ambignq_strategy, CoTAmbigNQStrategy)
    assert isinstance(fever_strategy, CoTFEVERStrategy)
