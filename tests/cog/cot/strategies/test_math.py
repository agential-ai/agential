"""Unit tests for CoT Math strategies."""

from agential.agent.cot.strategies.math import (
    CoTGSM8KStrategy,
    CoTSVAMPStrategy,
    CoTTabMWPStrategy,
)
from agential.llm.llm import MockLLM


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = CoTGSM8KStrategy(llm=llm)
    svamp_strategy = CoTSVAMPStrategy(llm=llm)
    tabmwp_strategy = CoTTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, CoTGSM8KStrategy)
    assert isinstance(svamp_strategy, CoTSVAMPStrategy)
    assert isinstance(tabmwp_strategy, CoTTabMWPStrategy)
