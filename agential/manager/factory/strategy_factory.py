"""Strategy factory class."""

from typing import Any

from agential.base.strategies import BaseStrategy
from agential.manager.mapping.strategy_mapping import STRATEGIES


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
