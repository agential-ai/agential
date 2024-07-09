"""ReAct prompts and fewshot examples selector."""

from typing import Any, Dict

from agential.base.factory import BaseFactory
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.react.prompts import (
    REACT_INSTRUCTION_AMBIGNQ,
    REACT_INSTRUCTION_FEVER,
    REACT_INSTRUCTION_GSM8K,
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_HUMANEVAL,
    REACT_INSTRUCTION_MBPP,
    REACT_INSTRUCTION_SVAMP,
    REACT_INSTRUCTION_TABMWP,
    REACT_INSTRUCTION_TRIVIAQA,
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

REACT_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [FewShotType.REACT],
    Benchmarks.FEVER: [FewShotType.REACT],
    Benchmarks.TRIVIAQA: [FewShotType.REACT],
    Benchmarks.AMBIGNQ: [FewShotType.REACT],
    Benchmarks.GSM8K: [FewShotType.REACT],
    Benchmarks.SVAMP: [FewShotType.REACT],
    Benchmarks.TABMWP: [FewShotType.REACT],
    Benchmarks.HUMANEVAL: [FewShotType.REACT],
    Benchmarks.MBPP: [FewShotType.REACT],
}

REACT_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": REACT_INSTRUCTION_HOTPOTQA,
    },
    Benchmarks.FEVER: {
        "prompt": REACT_INSTRUCTION_FEVER,
    },
    Benchmarks.TRIVIAQA: {
        "prompt": REACT_INSTRUCTION_TRIVIAQA,
    },
    Benchmarks.AMBIGNQ: {
        "prompt": REACT_INSTRUCTION_AMBIGNQ,
    },
    Benchmarks.GSM8K: {
        "prompt": REACT_INSTRUCTION_GSM8K,
    },
    Benchmarks.SVAMP: {
        "prompt": REACT_INSTRUCTION_SVAMP,
    },
    Benchmarks.TABMWP: {
        "prompt": REACT_INSTRUCTION_TABMWP,
    },
    Benchmarks.HUMANEVAL: {
        "prompt": REACT_INSTRUCTION_HUMANEVAL,
    },
    Benchmarks.MBPP: {
        "prompt": REACT_INSTRUCTION_MBPP,
    },
}
REACT_FEWSHOTS: Dict[str, Dict] = {
    Benchmarks.HOTPOTQA: {},
    Benchmarks.FEVER: {},
    Benchmarks.TRIVIAQA: {},
    Benchmarks.AMBIGNQ: {},
    Benchmarks.GSM8K: {},
    Benchmarks.SVAMP: {},
    Benchmarks.TABMWP: {},
    Benchmarks.HUMANEVAL: {},
    Benchmarks.MBPP: {},
}
REACT_STRATEGIES = {
    Benchmarks.HOTPOTQA: ReActHotQAStrategy,
    Benchmarks.FEVER: ReActFEVERStrategy,
    Benchmarks.TRIVIAQA: ReActTriviaQAStrategy,
    Benchmarks.AMBIGNQ: ReActAmbigNQStrategy,
    Benchmarks.GSM8K: ReActGSM8KStrategy,
    Benchmarks.SVAMP: ReActSVAMPStrategy,
    Benchmarks.TABMWP: ReActTabMWPStrategy,
    Benchmarks.HUMANEVAL: ReActHEvalStrategy,
    Benchmarks.MBPP: ReActMBPPStrategy,
}


class ReActFactory(BaseFactory):
    """A factory class for creating instances of ReAct strategies and selecting prompts and few-shot examples."""

    @staticmethod
    def get_fewshots(
        benchmark: str, fewshot_type: str, **kwargs: Any
    ) -> Dict[str, str]:
        """Retrieve few-shot examples based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            fewshot_type (str): The benchmark few-shot type.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of few-shot examples.
        """
        if benchmark not in REACT_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for ReAct.")

        if fewshot_type not in REACT_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for ReAct."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]

        return {"examples": benchmark_fewshots}

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        if benchmark not in REACT_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ReAct.")

        return REACT_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> ReActBaseStrategy:
        """Returns an instance of the appropriate ReAct strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            ReActBaseStrategy: An instance of the appropriate ReAct strategy.
        """
        if benchmark not in REACT_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent ReAct")

        strategy = REACT_STRATEGIES[benchmark]
        return strategy(**kwargs)
