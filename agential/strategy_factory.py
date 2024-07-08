"""Strategy factory classes."""

from typing import Any

from agential.base.strategies import BaseStrategy
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

"""Main file for managing benchmark prompts/few-shot examples."""

from agential.fewshots.ambignq import (
    AMBIGNQ_FEWSHOT_EXAMPLES_COT,
    AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.fever import (
    FEVER_FEWSHOT_EXAMPLES_COT,
    FEVER_FEWSHOT_EXAMPLES_DIRECT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.humaneval import (
    HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.mbpp import (
    MBPP_FEWSHOT_EXAMPLES_COT,
    MBPP_FEWSHOT_EXAMPLES_POT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.svamp import (
    SVAMP_FEWSHOT_EXAMPLES_COT,
    SVAMP_FEWSHOT_EXAMPLES_POT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.tabmwp import (
    TABMWP_FEWSHOT_EXAMPLES_COT,
    TABMWP_FEWSHOT_EXAMPLES_POT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
)
from agential.fewshots.triviaqa import (
    TRIVIAQA_FEWSHOT_EXAMPLES_COT,
    TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
)


class Benchmarks:
    """Supported benchmarks."""

    # QA.
    HOTPOTQA = "hotpotqa"
    FEVER = "fever"
    TRIVIAQA = "triviaqa"
    AMBIGNQ = "ambignq"

    # Math.
    GSM8K = "gsm8k"
    SVAMP = "svamp"
    TABMWP = "tabmwp"

    # Code.
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


class FewShotType:
    """Few-shot types."""

    COT = "cot"
    DIRECT = "direct"
    REACT = "react"
    POT = "pot"


class Agents:
    """Supported agents."""

    REACT = "react"
    REFLEXION_COT = "reflexion_cot"
    REFLEXION_REACT = "reflexion_react"
    CRITIC = "critic"


BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: {
        FewShotType.COT: HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.FEVER: {
        FewShotType.COT: FEVER_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: FEVER_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: FEVER_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.TRIVIAQA: {
        FewShotType.COT: TRIVIAQA_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: TRIVIAQA_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.AMBIGNQ: {
        FewShotType.COT: AMBIGNQ_FEWSHOT_EXAMPLES_COT,
        FewShotType.DIRECT: AMBIGNQ_FEWSHOT_EXAMPLES_DIRECT,
        FewShotType.REACT: AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    },
    Benchmarks.GSM8K: {
        FewShotType.POT: GSM8K_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: GSM8K_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: GSM8K_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.SVAMP: {
        FewShotType.POT: SVAMP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: SVAMP_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: SVAMP_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.TABMWP: {
        FewShotType.POT: TABMWP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: TABMWP_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: TABMWP_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.HUMANEVAL: {
        FewShotType.POT: HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: HUMANEVAL_FEWSHOT_EXAMPLES_COT,
    },
    Benchmarks.MBPP: {
        FewShotType.POT: MBPP_FEWSHOT_EXAMPLES_POT,
        FewShotType.REACT: MBPP_FEWSHOT_EXAMPLES_REACT,
        FewShotType.COT: MBPP_FEWSHOT_EXAMPLES_COT,
    },
}

STRATEGIES = {
    Agents.REACT: {
        Benchmarks.HOTPOTQA: ReActHotQAStrategy,
        Benchmarks.FEVER: ReActFEVERStrategy,
        Benchmarks.TRIVIAQA: ReActTriviaQAStrategy,
        Benchmarks.AMBIGNQ: ReActAmbigNQStrategy,
        Benchmarks.GSM8K: ReActGSM8KStrategy,
        Benchmarks.SVAMP: ReActSVAMPStrategy,
        Benchmarks.TABMWP: ReActTabMWPStrategy,
        Benchmarks.HUMANEVAL: ReActHEvalStrategy,
        Benchmarks.MBPP: ReActMBPPStrategy,
    },
    Agents.REFLEXION_COT: {
        Benchmarks.HOTPOTQA: ReflexionCoTHotQAStrategy,
        Benchmarks.FEVER: ReflexionCoTFEVERStrategy,
        Benchmarks.TRIVIAQA: ReflexionCoTTriviaQAStrategy,
        Benchmarks.AMBIGNQ: ReflexionCoTAmbigNQStrategy,
        Benchmarks.GSM8K: ReflexionCoTGSM8KStrategy,
        Benchmarks.SVAMP: ReflexionCoTSVAMPStrategy,
        Benchmarks.TABMWP: ReflexionCoTTabMWPStrategy,
        Benchmarks.HUMANEVAL: ReflexionCoTHEvalStrategy,
        Benchmarks.MBPP: ReflexionCoTMBPPStrategy,
    },
    Agents.REFLEXION_REACT: {
        Benchmarks.HOTPOTQA: ReflexionReActHotQAStrategy,
        Benchmarks.FEVER: ReflexionReActFEVERStrategy,
        Benchmarks.TRIVIAQA: ReflexionReActTriviaQAStrategy,
        Benchmarks.AMBIGNQ: ReflexionReActAmbigNQStrategy,
        Benchmarks.GSM8K: ReflexionReActGSM8KStrategy,
        Benchmarks.SVAMP: ReflexionReActSVAMPStrategy,
        Benchmarks.TABMWP: ReflexionReActTabMWPStrategy,
        Benchmarks.HUMANEVAL: ReflexionReActHEvalStrategy,
        Benchmarks.MBPP: ReflexionReActMBPPStrategy,
    },
    Agents.CRITIC: {
        Benchmarks.HOTPOTQA: CritHotQAStrategy,
        Benchmarks.FEVER: CritFEVERStrategy,
        Benchmarks.TRIVIAQA: CritTriviaQAStrategy,
        Benchmarks.AMBIGNQ: CritAmbigNQStrategy,
        Benchmarks.GSM8K: CritGSM8KStrategy,
        Benchmarks.SVAMP: CritSVAMPStrategy,
        Benchmarks.TABMWP: CritTabMWPStrategy,
        Benchmarks.HUMANEVAL: CritHEvalCodeStrategy,
        Benchmarks.MBPP: CritMBPPCodeStrategy,
    },
}


def get_benchmark_fewshots(benchmark: str, fewshot_type: str) -> str:
    """Retrieve few-shot examples for a given benchmark and few-shot type.

    Available Benchmarks:
        - hotpotqa: Supports "cot", "direct", "react"
        - fever: Supports "cot", "direct", "react"
        - triviaqa: Supports "cot", "direct", "react"
        - ambignq: Supports "cot", "direct", "react"
        - gsm8k: Supports "pot", "cot", "react"
        - svamp: Supports "pot", "cot", "react"
        - tabmwp: Supports "pot", "cot", "react"
        - humaneval: Supports "pot", "cot", "react"
        - mbpp: Supports "pot", "cot", "react"

    Available Few-Shot Types:
        - "cot"
        - "direct"
        - "react"
        - "pot"

    Args:
        benchmark (str): The benchmark name.
        fewshot_type (str): The type of few-shot examples. It should be one of the predefined types in the FewShotType class.

    Returns:
        str: The few-shot examples corresponding to the given benchmark and type.
        If the benchmark or few-shot type is not found, returns a detailed error message.
    """
    if benchmark not in BENCHMARK_FEWSHOTS:
        raise ValueError(f"Benchmark '{benchmark}' not found.")

    examples = BENCHMARK_FEWSHOTS[benchmark].get(fewshot_type)
    if examples is None:
        raise ValueError(
            f"Few-shot type '{fewshot_type}' not found for benchmark '{benchmark}'."
        )

    return examples


class StrategyFactory:
    """A factory class for creating instances of different strategies based on the specified agent and benchmark."""

    @staticmethod
    def get_strategy(agent: str, benchmark: str, **strategy_kwargs: Any) -> BaseStrategy:
        """Returns an instance of the appropriate strategy based on the provided agent and benchmark.

        Available agents:
            - "react"
            - "reflexion_cot"
            - "reflexion_react"
            - "critic"

        Available benchmarks:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            agent (str): The agent type.
            benchmark (str): The benchmark name.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            BaseStrategy: An instance of the appropriate strategy.

        Raises:
            ValueError: If the agent or benchmark is unsupported.
        """
        if agent == Agents.REACT:
            if benchmark == Benchmarks.HOTPOTQA:
                return ReActHotQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TRIVIAQA:
                return ReActTriviaQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.AMBIGNQ:
                return ReActAmbigNQStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.FEVER:
                return ReActFEVERStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.GSM8K:
                return ReActGSM8KStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.SVAMP:
                return ReActSVAMPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TABMWP:
                return ReActTabMWPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.HUMANEVAL:
                return ReActHEvalStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.MBPP:
                return ReActMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(
                    f"Unsupported benchmark: {benchmark} for agent {agent}"
                )

        elif agent == Agents.REFLEXION_COT:
            if benchmark == Benchmarks.HOTPOTQA:
                return ReflexionCoTHotQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TRIVIAQA:
                return ReflexionCoTTriviaQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.AMBIGNQ:
                return ReflexionCoTAmbigNQStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.FEVER:
                return ReflexionCoTFEVERStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.GSM8K:
                return ReflexionCoTGSM8KStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.SVAMP:
                return ReflexionCoTSVAMPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TABMWP:
                return ReflexionCoTTabMWPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.HUMANEVAL:
                return ReflexionCoTHEvalStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.MBPP:
                return ReflexionCoTMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(
                    f"Unsupported benchmark: {benchmark} for agent {agent}"
                )

        elif agent == Agents.REFLEXION_REACT:
            if benchmark == Benchmarks.HOTPOTQA:
                return ReflexionReActHotQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TRIVIAQA:
                return ReflexionReActTriviaQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.AMBIGNQ:
                return ReflexionReActAmbigNQStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.FEVER:
                return ReflexionReActFEVERStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.GSM8K:
                return ReflexionReActGSM8KStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.SVAMP:
                return ReflexionReActSVAMPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TABMWP:
                return ReflexionReActTabMWPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.HUMANEVAL:
                return ReflexionReActHEvalStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.MBPP:
                return ReflexionReActMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(
                    f"Unsupported benchmark: {benchmark} for agent {agent}"
                )

        elif agent == Agents.CRITIC:
            if benchmark == Benchmarks.HOTPOTQA:
                return CritHotQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TRIVIAQA:
                return CritTriviaQAStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.AMBIGNQ:
                return CritAmbigNQStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.FEVER:
                return CritFEVERStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.GSM8K:
                return CritGSM8KStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.SVAMP:
                return CritSVAMPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.TABMWP:
                return CritTabMWPStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.HUMANEVAL:
                return CritHEvalCodeStrategy(**strategy_kwargs)
            elif benchmark == Benchmarks.MBPP:
                return CritMBPPCodeStrategy(**strategy_kwargs)
            else:
                raise ValueError(
                    f"Unsupported benchmark: {benchmark} for agent {agent}"
                )

        else:
            raise ValueError(f"Unsupported agent: {agent}")
