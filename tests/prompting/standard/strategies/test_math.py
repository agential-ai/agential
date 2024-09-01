"""Unit tests for standard prompting Math strategies."""

from agential.llm.llm import MockLLM
from agential.prompting.standard.strategies.math import (
    StandardGSM8KStrategy,
    StandardSVAMPStrategy,
    StandardTabMWPStrategy,
)


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = StandardGSM8KStrategy(llm=llm)
    svamp_strategy = StandardSVAMPStrategy(llm=llm)
    tabmwp_strategy = StandardTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, StandardGSM8KStrategy)
    assert isinstance(svamp_strategy, StandardSVAMPStrategy)
    assert isinstance(tabmwp_strategy, StandardTabMWPStrategy)
