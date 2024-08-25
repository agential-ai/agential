"""Unit tests for CRITIC general strategy."""


from agential.cog.critic.strategies.general import CriticGeneralStrategy
from agential.llm.llm import BaseLLM, MockLLM


def test_init() -> None:
    """Test initialization of the CRITIC general strategy."""
    strategy = CriticGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)