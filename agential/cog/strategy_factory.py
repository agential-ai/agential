"""Strategy factory class."""

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
from agential.manager.constants import Agents, Benchmarks

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
