"""Unit tests for ReAct selector & factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.react.prompts import REACT_INSTRUCTION_HOTPOTQA
from agential.cog.react.selector import ReActFactory
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
from agential.base.constants import Benchmarks


def test_react_strategy_factory_get_strategy() -> None:
    """Tests ReActStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReActHotQAStrategy,
    )
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReActTriviaQAStrategy,
    )
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReActAmbigNQStrategy,
    )
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        ReActFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReActGSM8KStrategy,
    )
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReActSVAMPStrategy,
    )
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReActTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReActHEvalStrategy,
    )
    assert isinstance(
        ReActFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        ReActMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent ReAct"
    ):
        ReActFactory.get_strategy("unknown", llm=llm)


def test_react_factory_get_fewshots() -> None:
    """Tests ReActFactory get_fewshots method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = ReActFactory.get_fewshots(benchmark)
    assert isinstance(result, dict)
    assert result == {}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for ReAct."
    ):
        ReActFactory.get_fewshots("unknown")


def test_react_factory_get_prompt() -> None:
    """Tests ReActFactory get_prompt method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = ReActFactory.get_prompt(benchmark)
    assert result == {"prompt": REACT_INSTRUCTION_HOTPOTQA}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for ReAct."
    ):
        ReActFactory.get_prompt("unknown")
