"""Unit tests for CRITIC selector & factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.critic.selector import CriticSelector, CriticStrategyFactory
from agential.cog.critic.strategies.code import (
    CritHEvalCodeStrategy,
    CritMBPPCodeStrategy,
)
from agential.cog.critic.strategies.math import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
)
from agential.cog.critic.strategies.qa import (
    CritAmbigNQStrategy,
    CritFEVERStrategy,
    CritHotQAStrategy,
    CritTriviaQAStrategy,
)
from agential.manager.constants import Benchmarks


def test_critic_strategy_factory_get_strategy() -> None:
    """Tests CriticStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        CritHotQAStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        CritTriviaQAStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        CritAmbigNQStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        CritFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        CritGSM8KStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        CritSVAMPStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        CritTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        CritHEvalCodeStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        CritMBPPCodeStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent Critic"
    ):
        CriticStrategyFactory.get_strategy("unknown", llm=llm)
