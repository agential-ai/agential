"""Unit tests for strategy factory class."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel


from agential.cog.react.strategies.code import ReActHEvalStrategy, ReActMBPPStrategy
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
from agential.cog.reflexion.strategies.code import (
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
)
from agential.cog.reflexion.strategies.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
)
from agential.cog.reflexion.strategies.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTTriviaQAStrategy,
    ReflexionReActAmbigNQStrategy,
    ReflexionReActFEVERStrategy,
    ReflexionReActHotQAStrategy,
    ReflexionReActTriviaQAStrategy,
)
from agential.manager.constants import Agents, Benchmarks
from agential.manager.strategy_factory import (
    StrategyFactory,
)


import pytest
from agential.manager.constants import Agents, Benchmarks
from agential.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.manager.strategy_factory import (
    ReactStrategyFactory,
    ReflexionCoTStrategyFactory,
    ReflexionReActStrategyFactory,
    CriticStrategyFactory
)

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
    with pytest.raises(ValueError, match="Unsupported benchmark: unknown for agent ReAct"):
        ReactStrategyFactory.get_strategy("unknown", llm=llm)


def test_reflexion_cot_strategy_factory_get_strategy() -> None:
    """Tests ReflexionCoTStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReflexionCoTHotQAStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReflexionCoTTriviaQAStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReflexionCoTAmbigNQStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        ReflexionCoTFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReflexionCoTGSM8KStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReflexionCoTSVAMPStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReflexionCoTTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReflexionCoTHEvalStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        ReflexionCoTMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(ValueError, match="Unsupported benchmark: unknown for agent ReflexionCoT"):
        ReflexionCoTStrategyFactory.get_strategy("unknown", llm=llm)


def test_reflexion_react_strategy_factory_get_strategy() -> None:
    """Tests ReflexionReActStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReflexionReActHotQAStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReflexionReActTriviaQAStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReflexionReActAmbigNQStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        ReflexionReActFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReflexionReActGSM8KStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReflexionReActSVAMPStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReflexionReActTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReflexionReActHEvalStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        ReflexionReActMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(ValueError, match="Unsupported benchmark: unknown for agent ReflexionReAct"):
        ReflexionReActStrategyFactory.get_strategy("unknown", llm=llm)


