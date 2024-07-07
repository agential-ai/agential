"""Strategy factory classes."""

from typing import Any, Dict

from agential.cog.critic.strategies.base import CriticBaseStrategy
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
from agential.cog.react.strategies.base import ReActBaseStrategy
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
from agential.cog.reflexion.strategies.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
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


class ReActStrategyFactory:
    """A factory class for creating instances of different ReAct strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(benchmark: str, **strategy_kwargs) -> ReActBaseStrategy:
            Returns an instance of the appropriate ReAct strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(mode: str, **strategy_kwargs: Any) -> ReActBaseStrategy:
        """Returns an instance of the appropriate ReAct strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (str): A string specifying the benchmark.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            ReActBaseStrategy: An instance of the appropriate ReAct strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if mode == "hotpotqa":
            return ReActHotQAStrategy(**strategy_kwargs)
        elif mode == "triviaqa":
            return ReActTriviaQAStrategy(**strategy_kwargs)
        elif mode == "ambignq":
            return ReActAmbigNQStrategy(**strategy_kwargs)
        elif mode == "fever":
            return ReActFEVERStrategy(**strategy_kwargs)
        elif mode["math"] == "gsm8k":
            return ReActGSM8KStrategy(**strategy_kwargs)
        elif mode["math"] == "svamp":
            return ReActSVAMPStrategy(**strategy_kwargs)
        elif mode["math"] == "tabmwp":
            return ReActTabMWPStrategy(**strategy_kwargs)
        elif mode["code"] == "humaneval":
            return ReActHEvalStrategy(**strategy_kwargs)
        elif mode["code"] == "mbpp":
            return ReActMBPPStrategy(**strategy_kwargs)
        else:
            raise ValueError(f"Unsupported benchmark: {mode}")


class ReflexionCoTStrategyFactory:
    """A factory class for creating instances of different ReflexionCoT strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> ReflexionCoTBaseStrategy:
            Returns an instance of the appropriate ReflexionCoT strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(
        mode: Dict[str, str], **strategy_kwargs: Any
    ) -> ReflexionCoTBaseStrategy:
        """Returns an instance of the appropriate ReflexionCoT strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            ReflexionCoTBaseStrategy: An instance of the appropriate ReflexionCoT strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if mode["qa"] == "hotpotqa":
            return ReflexionCoTHotQAStrategy(**strategy_kwargs)
        elif mode["qa"] == "triviaqa":
            return ReflexionCoTTriviaQAStrategy(**strategy_kwargs)
        elif mode["qa"] == "ambignq":
            return ReflexionCoTAmbigNQStrategy(**strategy_kwargs)
        elif mode["qa"] == "fever":
            return ReflexionCoTFEVERStrategy(**strategy_kwargs)
        elif mode["math"] == "gsm8k":
            return ReflexionCoTGSM8KStrategy(**strategy_kwargs)
        elif mode["math"] == "svamp":
            return ReflexionCoTSVAMPStrategy(**strategy_kwargs)
        elif mode["math"] == "tabmwp":
            return ReflexionCoTTabMWPStrategy(**strategy_kwargs)
        elif mode["code"] == "humaneval":
            return ReflexionCoTHEvalStrategy(**strategy_kwargs)
        elif mode["code"] == "mbpp":
            return ReflexionCoTMBPPStrategy(**strategy_kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class ReflexionReActStrategyFactory:
    """A factory class for creating instances of different ReflexionReAct strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> ReflexionReActBaseStrategy:
            Returns an instance of the appropriate ReflexionReAct strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(
        mode: Dict[str, str], **strategy_kwargs: Any
    ) -> ReflexionReActBaseStrategy:
        """Returns an instance of the appropriate ReflexionReAct strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            ReflexionReActBaseStrategy: An instance of the appropriate ReflexionReAct strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if "qa" in mode:
            if mode["qa"] == "hotpotqa":
                return ReflexionReActHotQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "triviaqa":
                return ReflexionReActTriviaQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "ambignq":
                return ReflexionReActAmbigNQStrategy(**strategy_kwargs)
            elif mode["qa"] == "fever":
                return ReflexionReActFEVERStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported QA benchmark: {mode['qa']}")
        elif "math" in mode:
            if mode["math"] == "gsm8k":
                return ReflexionReActGSM8KStrategy(**strategy_kwargs)
            elif mode["math"] == "svamp":
                return ReflexionReActSVAMPStrategy(**strategy_kwargs)
            elif mode["math"] == "tabmwp":
                return ReflexionReActTabMWPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Math benchmark: {mode['math']}")
        elif "code" in mode:
            if mode["code"] == "humaneval":
                return ReflexionReActHEvalStrategy(**strategy_kwargs)
            elif mode["code"] == "mbpp":
                return ReflexionReActMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class CriticStrategyFactory:
    """A factory class for creating instances of different CRITIC strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> CriticBaseStrategy:
            Returns an instance of the appropriate Critic strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(
        mode: Dict[str, str], **strategy_kwargs: Any
    ) -> CriticBaseStrategy:
        """Returns an instance of the appropriate Critic strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            CriticBaseStrategy: An instance of the appropriate Critic strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if "qa" in mode:
            if mode["qa"] == "hotpotqa":
                return CritHotQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "triviaqa":
                return CritTriviaQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "ambignq":
                return CritAmbigNQStrategy(**strategy_kwargs)
            elif mode["qa"] == "fever":
                return CritFEVERStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported QA benchmark: {mode['qa']}")
        elif "math" in mode:
            if mode["math"] == "gsm8k":
                return CritGSM8KStrategy(**strategy_kwargs)
            elif mode["math"] == "svamp":
                return CritSVAMPStrategy(**strategy_kwargs)
            elif mode["math"] == "tabmwp":
                return CritTabMWPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Math benchmark: {mode['math']}")
        elif "code" in mode:
            if mode["code"] == "humaneval":
                return CritHEvalCodeStrategy(**strategy_kwargs)
            elif mode["code"] == "mbpp":
                return CritMBPPCodeStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
