"""Strategy factory classes."""

from typing import Any

from agential.manager.strategy_mapping import STRATEGIES
from agential.manager.fewshot_mapping import BENCHMARK_FEWSHOTS
from agential.base.strategies import BaseStrategy


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
    def get_strategy(
        agent: str, benchmark: str, **strategy_kwargs: Any
    ) -> BaseStrategy:
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
        if agent not in STRATEGIES:
            raise ValueError(f"Unsupported agent: {agent}")

        agent_strategies = STRATEGIES[agent]

        if benchmark not in agent_strategies:
            raise ValueError(f"Unsupported benchmark: {benchmark} for agent {agent}")

        strategy = agent_strategies[benchmark]
        return strategy(**strategy_kwargs)
