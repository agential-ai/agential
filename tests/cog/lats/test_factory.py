"""Unit tests for LATS factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.constants import Benchmarks
from agential.cog.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.lats.factory import LATSFactory
from agential.cog.lats.prompts import (
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_HOTPOTQA,
    LATS_REFLECT_INSTRUCTION_HOTPOTQA,
    LATS_VALUE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.lats.strategies.qa import (
    LATSAmbigNQStrategy,
    LATSFEVERStrategy,
    LATSHotQAStrategy,
    LATSTriviaQAStrategy,
)
from agential.cog.lats.strategies.math import (
    LATSGSM8KStrategy,
    LATSSVAMPStrategy,
    LATSTabMWPStrategy,
)
from agential.cog.lats.strategies.code import (
    LATSHEvalStrategy,
    LATSMBPPStrategy,
)


def test_LATS_factory_get_strategy() -> None:
    """Tests LATSFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        LATSHotQAStrategy,
    )
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        LATSTriviaQAStrategy,
    )
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        LATSAmbigNQStrategy,
    )
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        LATSFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        LATSGSM8KStrategy,
    )
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        LATSSVAMPStrategy,
    )
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        LATSTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        LATSHEvalStrategy,
    )
    assert isinstance(
        LATSFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        LATSMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent LATS"
    ):
        LATSFactory.get_strategy("unknown", llm=llm)


def test_LATS_factory_get_fewshots() -> None:
    """Tests LATSFactory get_fewshots method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = LATSFactory.get_fewshots(benchmark, fewshot_type="react")
    assert isinstance(result, dict)
    assert result == {
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_REFLECT,
        "value_examples": HOTPOTQA_FEWSHOT_EXAMPLES_LATS_VALUE,
    }

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for LATS."
    ):
        LATSFactory.get_fewshots("unknown", fewshot_type="react")

    # Test unsupported fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for LATS."
    ):
        LATSFactory.get_fewshots("hotpotqa", fewshot_type="pot")


def test_LATS_factory_get_prompts() -> None:
    """Tests LATSFactory get_prompts method."""
    # Test valid input.
    benchmark = Benchmarks.HOTPOTQA
    result = LATSFactory.get_prompts(benchmark)
    assert result == {
        "prompt": LATS_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": LATS_REFLECT_INSTRUCTION_HOTPOTQA,
        "value_prompt": LATS_VALUE_INSTRUCTION_HOTPOTQA,
    }

    # Test unsupported benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for LATS."
    ):
        LATSFactory.get_prompts("unknown")
