"""ReAct prompts and fewshot examples selector."""

from typing import Any, Dict

from agential.base.factory import BaseFactory
from agential.cog.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.cog.self_refine.strategies.base import SelfRefineBaseStrategy
from agential.cog.self_refine.prompts import (
    SELF_REFINE_INSTRUCTION_GSM8K,
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)
from agential.cog.self_refine.strategies.math import SelfRefineGSM8KStrategy

SELF_REFINE_BENCHMARK_FEWSHOTS = {
    Benchmarks.HOTPOTQA: [],
    Benchmarks.FEVER: [],
    Benchmarks.TRIVIAQA: [],
    Benchmarks.AMBIGNQ: [],
    Benchmarks.GSM8K: [FewShotType.POT],
    Benchmarks.SVAMP: [],
    Benchmarks.TABMWP: [],
    Benchmarks.HUMANEVAL: [],
    Benchmarks.MBPP: [],
}

SELF_REFINE_PROMPTS = {
    Benchmarks.HOTPOTQA: {
        "prompt": "",
    },
    Benchmarks.FEVER: {
        "prompt": "",
    },
    Benchmarks.TRIVIAQA: {
        "prompt": "",
    },
    Benchmarks.AMBIGNQ: {
        "prompt": "",
    },
    Benchmarks.GSM8K: {
        "prompt": SELF_REFINE_INSTRUCTION_GSM8K,
        "critique_prompt": SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        "refine_prompt": SELF_REFINE_REFINE_INSTRUCTION_GSM8K
    },
    Benchmarks.SVAMP: {
        "prompt": "",
    },
    Benchmarks.TABMWP: {
        "prompt": "",
    },
    Benchmarks.HUMANEVAL: {
        "prompt": "",
    },
    Benchmarks.MBPP: {
        "prompt": "",
    },
}

SELF_REFINE_FEWSHOTS: Dict[str, Dict] = {
    Benchmarks.HOTPOTQA: {},
    Benchmarks.FEVER: {},
    Benchmarks.TRIVIAQA: {},
    Benchmarks.AMBIGNQ: {},
    Benchmarks.GSM8K: {
        "critique_examples": GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        "refine_examples": GSM8K_REFINE_FEWSHOT_EXAMPLES
    },
    Benchmarks.SVAMP: {},
    Benchmarks.TABMWP: {},
    Benchmarks.HUMANEVAL: {},
    Benchmarks.MBPP: {},
}

SELF_REFINE_STRATEGIES = {
    Benchmarks.HOTPOTQA: None,
    Benchmarks.FEVER: None,
    Benchmarks.TRIVIAQA: None,
    Benchmarks.AMBIGNQ: None,
    Benchmarks.GSM8K: SelfRefineGSM8KStrategy,
    Benchmarks.SVAMP: None,
    Benchmarks.TABMWP: None,
    Benchmarks.HUMANEVAL: None,
    Benchmarks.MBPP: None,
}

class SelfRefineFactory(BaseFactory):
    """A factory class for creating instances of Self-Refine strategies and selecting prompts and few-shot examples."""

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
        if benchmark not in SELF_REFINE_FEWSHOTS:
            raise ValueError(f"Benchmark '{benchmark}' few-shots not found for Self-Refine.")

        if fewshot_type not in SELF_REFINE_BENCHMARK_FEWSHOTS[benchmark]:
            raise ValueError(
                f"Benchmark '{benchmark}' few-shot type not supported for Self-Refine."
            )

        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark]

        return {
            "examples": benchmark_fewshots,
            **SELF_REFINE_FEWSHOTS[benchmark]
        }

    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Retrieve the prompt instruction based on the benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional arguments.

        Returns:
            Dict[str, str]: A dictionary of prompt instructions.
        """
        if benchmark not in SELF_REFINE_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for Self-Refine.")

        return SELF_REFINE_PROMPTS[benchmark]

    @staticmethod
    def get_strategy(benchmark: str, **kwargs: Any) -> SelfRefineBaseStrategy:
        """Returns an instance of the appropriate Self-Refine strategy based on the provided benchmark.

        Args:
            benchmark (str): The benchmark name.
            **kwargs (Any): Additional keyword arguments to pass to
                the strategy's constructor.

        Returns:
            SelfRefineBaseStrategy: An instance of the appropriate Self-Refine strategy.
        """
        if benchmark not in SELF_REFINE_STRATEGIES:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent Self-Refine")

        strategy = SELF_REFINE_STRATEGIES[benchmark]
        if strategy is None:
            raise ValueError(f"No strategy defined for benchmark: {benchmark}")

        return strategy(**kwargs)
