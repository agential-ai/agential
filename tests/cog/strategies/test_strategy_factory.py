"""Unit tests for strategy factory classes."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.strategies.critic.code_strategy import (
    CritHEvalCodeStrategy,
    CritMBPPCodeStrategy,
)
from agential.cog.strategies.critic.math_strategy import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
)
from agential.cog.strategies.critic.qa_strategy import (
    CritAmbigNQStrategy,
    CritFEVERStrategy,
    CritHotQAStrategy,
    CritTriviaQAStrategy,
)
from agential.cog.strategies.strategy_factory import CriticStrategyFactory


def test_critic_strategy_factory_get_strategy() -> None:
    """Tests CriticStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "hotpotqa"}, llm=llm),
        CritHotQAStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "triviaqa"}, llm=llm),
        CritTriviaQAStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "ambignq"}, llm=llm),
        CritAmbigNQStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "fever"}, llm=llm), CritFEVERStrategy
    )

    # Math benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy({"math": "gsm8k"}, llm=llm),
        CritGSM8KStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"math": "svamp"}, llm=llm),
        CritSVAMPStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"math": "tabmwp"}, llm=llm),
        CritTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy({"code": "mbpp"}, llm=llm),
        CritMBPPCodeStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"code": "humaneval"}, llm=llm),
        CritHEvalCodeStrategy,
    )

    # Unsupported benchmarks.
    with pytest.raises(ValueError, match="Unsupported QA benchmark: unknown"):
        CriticStrategyFactory.get_strategy({"qa": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Math benchmark: unknown"):
        CriticStrategyFactory.get_strategy({"math": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Code benchmark: unknown"):
        CriticStrategyFactory.get_strategy({"code": "unknown"})

    with pytest.raises(ValueError, match="Unsupported mode: {}"):
        CriticStrategyFactory.get_strategy({})
