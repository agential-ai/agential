"""Unit tests for ReAct factory."""

import pytest

from agential.llm.llm import MockLLM

from agential.cog.constants import Benchmarks
from agential.cog.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.react.factory import ReActFactory
from agential.cog.react.prompts import REACT_INSTRUCTION_HOTPOTQA
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


def test_react_factory_get_strategy() -> None:
    """Tests ReActFactory get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

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
    result = ReActFactory.get_fewshots(benchmark, fewshot_type="react")
    assert isinstance(result, dict)
    assert result == {"examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for ReAct."
    ):
        ReActFactory.get_fewshots("unknown", fewshot_type="react")

    # Test unsupported fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for ReAct."
    ):
        ReActFactory.get_fewshots("hotpotqa", fewshot_type="pot")


def test_react_factory_get_prompts() -> None:
    """Tests ReActFactory get_prompts method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = ReActFactory.get_prompts(benchmark)
    assert result == {"prompt": REACT_INSTRUCTION_HOTPOTQA}

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for ReAct."
    ):
        ReActFactory.get_prompts("unknown")
