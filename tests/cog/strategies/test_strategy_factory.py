"""Unit tests for strategy factory classes."""

from unittest.mock import MagicMock

import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

from agential.cog.strategies.critic.code import (
    CritHEvalCodeStrategy,
    CritMBPPCodeStrategy,
)
from agential.cog.strategies.critic.math import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
)
from agential.cog.strategies.critic.qa import (
    CritAmbigNQStrategy,
    CritFEVERStrategy,
    CritHotQAStrategy,
    CritTriviaQAStrategy,
)
from agential.cog.strategies.strategy_factory import (
    CriticStrategyFactory, 
    SelfRefineStrategyFactory
)
from agential.cog.strategies.self_refine.math import (
    SelfRefineGSM8KStrategy
)


def test_critic_strategy_factory_get_strategy() -> None:
    """Tests CriticStrategyFactory get_strategy method."""
    search = MagicMock(spec=GoogleSerperAPIWrapper)

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

    # Test kwargs for QA strategy.
    strategy = CriticStrategyFactory.get_strategy(
        {"qa": "fever"}, llm=llm, search=search, evidence_length=500, num_results=10
    )
    assert isinstance(strategy, CritFEVERStrategy)
    assert strategy.llm == llm
    assert isinstance(strategy.search, GoogleSerperAPIWrapper)
    assert strategy.evidence_length == 500
    assert strategy.num_results == 10

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

    # Test kwargs for Math strategy.
    strategy = CriticStrategyFactory.get_strategy(
        {"math": "gsm8k"}, llm=llm, patience=3
    )
    assert isinstance(strategy, CritGSM8KStrategy)
    assert strategy.llm == llm
    assert strategy.patience == 3

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


def test_self_refine_strategy_factory_get_strategy() -> None:
    """Tests SelfRefineStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # Math benchmarks.
    assert isinstance(
        SelfRefineStrategyFactory.get_strategy({"math": "gsm8k"}, llm=llm),
        SelfRefineGSM8KStrategy,
    )

    # Test kwargs for Math strategy.
    strategy = SelfRefineStrategyFactory.get_strategy(
        {"math": "gsm8k"}, llm=llm, patience=3
    )
    assert isinstance(strategy, SelfRefineGSM8KStrategy)
    assert strategy.llm == llm
    assert strategy.patience == 3
    
    # Unsupported benchmarks.
    with pytest.raises(ValueError, match="Unsupported QA benchmark: unknown"):
        SelfRefineStrategyFactory.get_strategy({"qa": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Math benchmark: unknown"):
        SelfRefineStrategyFactory.get_strategy({"math": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Code benchmark: unknown"):
        SelfRefineStrategyFactory.get_strategy({"code": "unknown"})

    with pytest.raises(ValueError, match="Unsupported mode: {}"):
        SelfRefineStrategyFactory.get_strategy({})
