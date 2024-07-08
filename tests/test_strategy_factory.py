"""Unit tests for strategy factory classes."""

from unittest.mock import MagicMock

import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

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
from agential.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.manager.strategy_factory import (
    StrategyFactory,
    get_benchmark_fewshots,
)
from agential.manager.constants import Agents, Benchmarks, FewShotType


def test_strategy_factory_get_strategy() -> None:
    """Tests StrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks for ReAct agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.HOTPOTQA, llm=llm),
        ReActHotQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.TRIVIAQA, llm=llm),
        ReActTriviaQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.AMBIGNQ, llm=llm),
        ReActAmbigNQStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.FEVER, llm=llm),
        ReActFEVERStrategy,
    )

    # Math benchmarks for ReAct agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.GSM8K, llm=llm),
        ReActGSM8KStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.SVAMP, llm=llm),
        ReActSVAMPStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.TABMWP, llm=llm),
        ReActTabMWPStrategy,
    )

    # Code benchmarks for ReAct agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.HUMANEVAL, llm=llm),
        ReActHEvalStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REACT, Benchmarks.MBPP, llm=llm),
        ReActMBPPStrategy,
    )

    # QA benchmarks for ReflexionCoT agent.
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_COT, Benchmarks.HOTPOTQA, llm=llm
        ),
        ReflexionCoTHotQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_COT, Benchmarks.TRIVIAQA, llm=llm
        ),
        ReflexionCoTTriviaQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, Benchmarks.AMBIGNQ, llm=llm),
        ReflexionCoTAmbigNQStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, Benchmarks.FEVER, llm=llm),
        ReflexionCoTFEVERStrategy,
    )

    # Math benchmarks for ReflexionCoT agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, Benchmarks.GSM8K, llm=llm),
        ReflexionCoTGSM8KStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, Benchmarks.SVAMP, llm=llm),
        ReflexionCoTSVAMPStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, Benchmarks.TABMWP, llm=llm),
        ReflexionCoTTabMWPStrategy,
    )

    # Code benchmarks for ReflexionCoT agent.
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_COT, Benchmarks.HUMANEVAL, llm=llm
        ),
        ReflexionCoTHEvalStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, Benchmarks.MBPP, llm=llm),
        ReflexionCoTMBPPStrategy,
    )

    # QA benchmarks for ReflexionReAct agent.
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_REACT, Benchmarks.HOTPOTQA, llm=llm
        ),
        ReflexionReActHotQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_REACT, Benchmarks.TRIVIAQA, llm=llm
        ),
        ReflexionReActTriviaQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_REACT, Benchmarks.AMBIGNQ, llm=llm
        ),
        ReflexionReActAmbigNQStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_REACT, Benchmarks.FEVER, llm=llm),
        ReflexionReActFEVERStrategy,
    )

    # Math benchmarks for ReflexionReAct agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_REACT, Benchmarks.GSM8K, llm=llm),
        ReflexionReActGSM8KStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_REACT, Benchmarks.SVAMP, llm=llm),
        ReflexionReActSVAMPStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_REACT, Benchmarks.TABMWP, llm=llm
        ),
        ReflexionReActTabMWPStrategy,
    )

    # Code benchmarks for ReflexionReAct agent.
    assert isinstance(
        StrategyFactory.get_strategy(
            Agents.REFLEXION_REACT, Benchmarks.HUMANEVAL, llm=llm
        ),
        ReflexionReActHEvalStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.REFLEXION_REACT, Benchmarks.MBPP, llm=llm),
        ReflexionReActMBPPStrategy,
    )

    # QA benchmarks for Critic agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.HOTPOTQA, llm=llm),
        CritHotQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.TRIVIAQA, llm=llm),
        CritTriviaQAStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.AMBIGNQ, llm=llm),
        CritAmbigNQStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.FEVER, llm=llm),
        CritFEVERStrategy,
    )

    # Math benchmarks for Critic agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.GSM8K, llm=llm),
        CritGSM8KStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.SVAMP, llm=llm),
        CritSVAMPStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.TABMWP, llm=llm),
        CritTabMWPStrategy,
    )

    # Code benchmarks for Critic agent.
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.HUMANEVAL, llm=llm),
        CritHEvalCodeStrategy,
    )
    assert isinstance(
        StrategyFactory.get_strategy(Agents.CRITIC, Benchmarks.MBPP, llm=llm),
        CritMBPPCodeStrategy,
    )

    # Unsupported benchmarks.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent react"
    ):
        StrategyFactory.get_strategy(Agents.REACT, "unknown", llm=llm)

    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent reflexion_cot"
    ):
        StrategyFactory.get_strategy(Agents.REFLEXION_COT, "unknown", llm=llm)

    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent reflexion_react"
    ):
        StrategyFactory.get_strategy(Agents.REFLEXION_REACT, "unknown", llm=llm)

    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent critic"
    ):
        StrategyFactory.get_strategy(Agents.CRITIC, "unknown", llm=llm)

    with pytest.raises(ValueError, match="Unsupported agent: unknown"):
        StrategyFactory.get_strategy("unknown", Benchmarks.HOTPOTQA, llm=llm)


def test_get_benchmark_fewshots() -> None:
    """Test get_benchmark_fewshots."""
    # Test valid input.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.COT
    result = get_benchmark_fewshots(benchmark, fewshot_type)
    assert result == HOTPOTQA_FEWSHOT_EXAMPLES_COT

    # Test invalid benchmark.
    benchmark = "invalid_benchmark"
    fewshot_type = FewShotType.COT
    with pytest.raises(ValueError):
        result = get_benchmark_fewshots(benchmark, fewshot_type)

    # Test invalid few-shot type.
    benchmark = "hotpotqa"
    fewshot_type = "invalid_fewshot"
    with pytest.raises(ValueError):
        result = get_benchmark_fewshots(benchmark, fewshot_type)

    # Test invalid few-shot type for the given benchmark.
    benchmark = "hotpotqa"
    fewshot_type = FewShotType.POT
    with pytest.raises(ValueError):
        result = get_benchmark_fewshots(benchmark, fewshot_type)
