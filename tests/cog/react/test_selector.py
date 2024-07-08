"""Unit tests for ReAct selector & factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.react.selector import ReActSelector, ReactStrategyFactory
from agential.cog.react.strategies.code import (
    ReActHEvalStrategy,
    ReActMBPPStrategy,
)
from agential.cog.react.strategies.math import (
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
)
from agential.cog.react.strategies.qa import (
    ReActAmbigNQStrategy,
    ReActFEVERStrategy,
    ReActHotQAStrategy,
    ReActTriviaQAStrategy,
)
from agential.manager.constants import Benchmarks


def test_react_strategy_factory_get_strategy() -> None:
    """Tests ReActStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReActHotQAStrategy,
    )
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReActTriviaQAStrategy,
    )
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReActAmbigNQStrategy,
    )
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        ReActFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReActGSM8KStrategy,
    )
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReActSVAMPStrategy,
    )
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReActTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReActHEvalStrategy,
    )
    assert isinstance(
        ReactStrategyFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        ReActMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent ReAct"
    ):
        ReactStrategyFactory.get_strategy("unknown", llm=llm)
