"""Unit tests for CRITIC general strategy."""


from agential.cog.critic.strategies.general import CriticGeneralStrategy
from agential.llm.llm import MockLLM


def test_init() -> None:
    """Test initialization of the CRITIC general strategy."""
    strategy = CriticGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert strategy.llm == MockLLM("gpt-3.5-turbo", responses=[])