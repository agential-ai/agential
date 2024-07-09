"""Unit tests for CRITIC factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.constants import Benchmarks
from agential.cog.critic.factory import CriticFactory
from agential.cog.critic.prompts import (
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_GSM8K,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
)
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
from agential.cog.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT


def test_critic_factory_get_strategy() -> None:
    """Tests CriticFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.HOTPOTQA, llm=llm),
        CritHotQAStrategy,
    )
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.TRIVIAQA, llm=llm),
        CritTriviaQAStrategy,
    )
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.AMBIGNQ, llm=llm),
        CritAmbigNQStrategy,
    )
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.FEVER, llm=llm),
        CritFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.GSM8K, llm=llm),
        CritGSM8KStrategy,
    )
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.SVAMP, llm=llm),
        CritSVAMPStrategy,
    )
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.TABMWP, llm=llm),
        CritTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.HUMANEVAL, llm=llm),
        CritHEvalCodeStrategy,
    )
    assert isinstance(
        CriticFactory.get_strategy(Benchmarks.MBPP, llm=llm),
        CritMBPPCodeStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent Critic"
    ):
        CriticFactory.get_strategy("unknown", llm=llm)


def test_critic_factory_get_fewshots() -> None:
    """Tests CriticFactory get_fewshots method."""
    # Valid benchmark with tool usage.
    benchmark = Benchmarks.GSM8K
    fewshots = CriticFactory.get_fewshots(benchmark, fewshot_type="pot", use_tool=True)
    assert "critique_examples" in fewshots
    assert fewshots == {"examples": GSM8K_FEWSHOT_EXAMPLES_POT, "critique_examples": GSM8K_FEWSHOT_EXAMPLES_CRITIC}

    # Valid benchmark without tool usage.
    fewshots = CriticFactory.get_fewshots(benchmark, fewshot_type="pot", use_tool=False)
    assert "critique_examples" in fewshots
    assert fewshots == {
        "examples": GSM8K_FEWSHOT_EXAMPLES_POT,
        "critique_examples": GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    }

    # Invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for Critic."
    ):
        CriticFactory.get_fewshots("unknown", fewshot_type="pot", use_tool=True)

    # Invalid fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for Critic."
    ):
        CriticFactory.get_fewshots("hotpotqa", fewshot_type="pot", use_tool=True)

    # Missing use_tool argument.
    with pytest.raises(ValueError, match="`use_tool` not specified."):
        CriticFactory.get_fewshots(benchmark, fewshot_type="pot")


def test_critic_factory_get_prompts() -> None:
    """Tests CriticFactory get_prompts method."""
    # Valid benchmark with tool usage.
    benchmark = Benchmarks.GSM8K
    prompts = CriticFactory.get_prompts(benchmark, use_tool=True)
    assert "prompt" in prompts
    assert "critique_prompt" in prompts
    assert prompts == {
        "prompt": CRITIC_POT_INSTRUCTION_GSM8K,
        "critique_prompt": CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    }

    # Valid benchmark without tool usage.
    prompts = CriticFactory.get_prompts(benchmark, use_tool=False)
    assert "prompt" in prompts
    assert "critique_prompt" in prompts
    assert prompts == {
        "prompt": CRITIC_POT_INSTRUCTION_GSM8K,
        "critique_prompt": CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    }

    # Invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for Critic."
    ):
        CriticFactory.get_prompts("unknown", use_tool=True)

    # Missing use_tool argument.
    with pytest.raises(ValueError, match="`use_tool` not specified."):
        CriticFactory.get_prompts(benchmark)
