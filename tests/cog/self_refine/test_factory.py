"""Unit tests for Reflexion factory."""

import pytest

from agential.llm.llm import MockLLM

from agential.cog.constants import Benchmarks, FewShotType
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_COT
from agential.cog.self_refine.factory import (
    SelfRefineFactory,
)
from agential.cog.self_refine.prompts import (
    HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
    HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_INSTRUCTION_HOTPOTQA,
    SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
)
from agential.cog.self_refine.strategies.math import (
    SelfRefineGSM8KStrategy,
    SelfRefineSVAMPStrategy,
    SelfRefineTabMWPStrategy,
)
from agential.cog.self_refine.strategies.qa import (
    SelfRefineAmbigNQStrategy,
    SelfRefineFEVERStrategy,
    SelfRefineHotQAStrategy,
    SelfRefineTriviaQAStrategy,
)


def test_self_refine_factory_get_strategy() -> None:
    """Tests SelfRefineFactory get_strategy method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # QA benchmarks.
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        SelfRefineHotQAStrategy,
    )
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        SelfRefineTriviaQAStrategy,
    )
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        SelfRefineAmbigNQStrategy,
    )
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        SelfRefineFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        SelfRefineGSM8KStrategy,
    )
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        SelfRefineSVAMPStrategy,
    )
    assert isinstance(
        SelfRefineFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        SelfRefineTabMWPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent Self-Refine"
    ):
        SelfRefineFactory.get_strategy("unknown", llm=llm)


def test_self_refine_factory_get_fewshots() -> None:
    """Tests SelfRefineFactory get_fewshots method."""
    # Test with valid fewshot type.
    fewshots = SelfRefineFactory.get_fewshots(Benchmarks.HOTPOTQA, FewShotType.COT)
    assert isinstance(fewshots, dict)
    assert "examples" in fewshots
    assert "critique_examples" in fewshots
    assert "refine_examples" in fewshots
    assert fewshots == {
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        "critique_examples": HOTPOTQA_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": HOTPOTQA_REFINE_FEWSHOT_EXAMPLES,
    }

    # Test with invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for Self-Refine."
    ):
        SelfRefineFactory.get_fewshots("unknown", FewShotType.COT)

    # Test with invalid fewshot type.
    with pytest.raises(
        ValueError,
        match="Benchmark 'hotpotqa' few-shot type not supported for Self-Refine.",
    ):
        SelfRefineFactory.get_fewshots(Benchmarks.HOTPOTQA, "invalid_type")


def test_self_refine_factory_get_prompts() -> None:
    """Tests SelfRefineFactory get_prompts method."""
    # Test with valid benchmark.
    prompts = SelfRefineFactory.get_prompts(Benchmarks.HOTPOTQA)
    assert isinstance(prompts, dict)
    assert "prompt" in prompts
    assert "critique_prompt" in prompts
    assert "refine_prompt" in prompts
    assert prompts == {
        "prompt": SELF_REFINE_INSTRUCTION_HOTPOTQA,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_HOTPOTQA,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_HOTPOTQA,
    }

    # Test with invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for Self-Refine."
    ):
        SelfRefineFactory.get_prompts("unknown")
