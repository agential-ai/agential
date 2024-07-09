"""Unit tests for Reflexion factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.base.constants import Benchmarks
from agential.cog.reflexion.factory import (
    ReflexionCoTFactory,
    ReflexionReActFactory,
)
from agential.cog.reflexion.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
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


def test_reflexion_cot_factory_get_strategy() -> None:
    """Tests ReflexionCoTFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReflexionCoTHotQAStrategy,
    )
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReflexionCoTTriviaQAStrategy,
    )
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReflexionCoTAmbigNQStrategy,
    )
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        ReflexionCoTFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReflexionCoTGSM8KStrategy,
    )
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReflexionCoTSVAMPStrategy,
    )
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReflexionCoTTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReflexionCoTHEvalStrategy,
    )
    assert isinstance(
        ReflexionCoTFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        ReflexionCoTMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent ReflexionCoT"
    ):
        ReflexionCoTFactory.get_strategy("unknown", llm=llm)


def test_reflexion_react_factory_get_strategy() -> None:
    """Tests ReflexionReActFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        ReflexionReActHotQAStrategy,
    )
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        ReflexionReActTriviaQAStrategy,
    )
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        ReflexionReActAmbigNQStrategy,
    )
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        ReflexionReActFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        ReflexionReActGSM8KStrategy,
    )
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        ReflexionReActSVAMPStrategy,
    )
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        ReflexionReActTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        ReflexionReActHEvalStrategy,
    )
    assert isinstance(
        ReflexionReActFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        ReflexionReActMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent ReflexionReAct"
    ):
        ReflexionReActFactory.get_strategy("unknown", llm=llm)


def test_reflexion_cot_factory_get_fewshots() -> None:
    """Tests ReflexionCoTFactory get_fewshots method."""
    # Valid benchmark.
    benchmark = Benchmarks.HOTPOTQA
    fewshots = ReflexionCoTFactory.get_fewshots(benchmark)
    assert isinstance(fewshots, dict)
    assert fewshots == {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    }

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for ReflexionCoT."
    ):
        ReflexionCoTFactory.get_fewshots("unknown")


def test_reflexion_cot_factory_get_prompt() -> None:
    """Tests ReflexionCoTFactory get_prompt method."""
    # Valid benchmark.
    benchmark = Benchmarks.HOTPOTQA
    prompt = ReflexionCoTFactory.get_prompt(benchmark)
    assert isinstance(prompt, dict)
    assert prompt == {
        "prompt": REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    }

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for ReflexionCoT."
    ):
        ReflexionCoTFactory.get_prompt("unknown")


def test_reflexion_react_factory_get_fewshots() -> None:
    """Tests ReflexionReActFactory get_fewshots method."""
    # Valid benchmark.
    benchmark = Benchmarks.HOTPOTQA
    fewshots = ReflexionReActFactory.get_fewshots(benchmark)
    assert isinstance(fewshots, dict)
    assert fewshots == {
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    }

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for ReflexionReAct."
    ):
        ReflexionReActFactory.get_fewshots("unknown")


def test_reflexion_react_factory_get_prompt() -> None:
    """Tests ReflexionReActFactory get_prompt method."""
    # Valid benchmark.
    benchmark = Benchmarks.HOTPOTQA
    prompt = ReflexionReActFactory.get_prompt(benchmark)
    assert isinstance(prompt, dict)
    assert prompt == {
        "prompt": REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    }

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for ReflexionReAct."
    ):
        ReflexionReActFactory.get_prompt("unknown")
