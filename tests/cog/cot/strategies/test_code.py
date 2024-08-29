"""Unit tests for CoT Code strategies."""


from agential.cog.cot.strategies.code import (
    CoTMBPPCodeStrategy,
    CoTHEvalCodeStrategy,
)
from agential.llm.llm import MockLLM


def test_instantiate_code_strategies() -> None:
    """Test instantiate all Code strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    mbpp_strategy = CoTMBPPCodeStrategy(llm=llm)
    heval_strategy = CoTHEvalCodeStrategy(llm=llm)

    assert isinstance(mbpp_strategy, CoTMBPPCodeStrategy)
    assert isinstance(heval_strategy, CoTHEvalCodeStrategy)
