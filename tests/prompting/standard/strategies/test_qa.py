"""Unit tests for standard prompting QA strategies."""

from agential.llm.llm import MockLLM
from agential.prompting.standard.strategies.qa import (
    StandardAmbigNQStrategy,
    StandardFEVERStrategy,
    StandardHotQAStrategy,
    StandardTriviaQAStrategy,
)


def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    hotqa_strategy = StandardHotQAStrategy(llm=llm)
    triviaqa_strategy = StandardTriviaQAStrategy(llm=llm)
    ambignq_strategy = StandardAmbigNQStrategy(llm=llm)
    fever_strategy = StandardFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, StandardHotQAStrategy)
    assert isinstance(triviaqa_strategy, StandardTriviaQAStrategy)
    assert isinstance(ambignq_strategy, StandardAmbigNQStrategy)
    assert isinstance(fever_strategy, StandardFEVERStrategy)
