"""Unit tests for CoT Code strategies."""


from agential.cog.cot.strategies.code import (
    CoTMBPPStrategy,
    CoTHEvalStrategy,
)
from agential.llm.llm import MockLLM


def test_instantiate_code_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    mbpp_strategy = CoTMBPPStrategy(llm=llm)
    heval_strategy = CoTHEvalStrategy(llm=llm)

    assert isinstance(mbpp_strategy, CoTMBPPStrategy)
    assert isinstance(heval_strategy, CoTHEvalStrategy)
