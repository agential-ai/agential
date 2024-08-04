"""Unit tests for ExpeL factory."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.constants import Benchmarks
from agential.cog.expel.factory import (
    ExpeLFactory,
)
from agential.cog.expel.prompts import (
    EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT
)
from agential.cog.expel.strategies.code import (
    ExpeLHEvalStrategy,
    ExpeLMBPPStrategy,
)
from agential.cog.expel.strategies.math import (
    ExpeLGSM8KStrategy,
    ExpeLSVAMPStrategy,
    ExpeLTabMWPStrategy,
)
from agential.cog.expel.strategies.qa import (
    ExpeLAmbigNQStrategy,
    ExpeLFEVERStrategy,
    ExpeLHotQAStrategy,
    ExpeLTriviaQAStrategy,
)
from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.reflexion.agent import ReflexionReActAgent


def test_expel_factory_get_strategy() -> None:
    """Tests ExpeLFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.HOTPOTQA,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.HOTPOTQA
            ),
        ),
        ExpeLHotQAStrategy,
    )
    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.TRIVIAQA,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.TRIVIAQA
            ),
        ),
        ExpeLTriviaQAStrategy,
    )
    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.AMBIGNQ,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.AMBIGNQ
            ),
        ),
        ExpeLAmbigNQStrategy,
    )
    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.FEVER,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.FEVER
            ),
        ),
        ExpeLFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.GSM8K,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.GSM8K
            ),
        ),
        ExpeLGSM8KStrategy,
    )

    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.SVAMP,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.SVAMP
            ),
        ),
        ExpeLSVAMPStrategy,
    )

    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.TABMWP,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.TABMWP
            ),
        ),
        ExpeLTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.HUMANEVAL,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.HUMANEVAL
            ),
        ),
        ExpeLHEvalStrategy,
    )

    assert isinstance(
        ExpeLFactory.get_strategy(
            Benchmarks.MBPP,
            llm=llm,
            reflexion_react_agent=ReflexionReActAgent(
                llm=llm, benchmark=Benchmarks.MBPP
            ),
        ),
        ExpeLMBPPStrategy,
    )

    # Unsupported benchmark.
    with pytest.raises(
        ValueError, match="Unsupported benchmark: unknown for agent ExpeL"
    ):
        ExpeLFactory.get_strategy("unknown", llm=llm)


def test_expel_factory_get_fewshots() -> None:
    """Tests ExpeLFactory get_fewshots method."""
    # Valid benchmark.
    benchmark = Benchmarks.HOTPOTQA
    fewshots = ExpeLFactory.get_fewshots(benchmark, fewshot_type="react")
    assert "reflect_examples" in fewshots
    assert fewshots == {
        "examples": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_examples": HOTPOTQA_FEWSHOT_EXAMPLES_EXPEL_REFLEXION_REACT_REFLECT,
    }

    # Invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' few-shots not found for ExpeL."
    ):
        ExpeLFactory.get_fewshots("unknown", fewshot_type="react")

    # Invalid fewshot_type.
    with pytest.raises(
        ValueError, match="Benchmark 'hotpotqa' few-shot type not supported for ExpeL."
    ):
        ExpeLFactory.get_fewshots("hotpotqa", fewshot_type="pot")


def test_expel_factory_get_prompts() -> None:
    """Tests ExpeLFactory get_prompts method."""
    # Valid benchmark.
    benchmark = Benchmarks.HOTPOTQA
    prompts = ExpeLFactory.get_prompts(benchmark)
    assert "prompt" in prompts
    assert "reflect_prompt" in prompts
    assert prompts == {
        "prompt": EXPEL_REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "reflect_prompt": EXPEL_REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    }

    # Invalid benchmark.
    with pytest.raises(
        ValueError, match="Benchmark 'unknown' prompt not found for ExpeL."
    ):
        ExpeLFactory.get_prompts("unknown")
