"""Strategy factory classes."""

from typing import Dict

from agential.cog.strategies.critic.base import CriticBaseStrategy
from agential.cog.strategies.critic.code_strategy import (
    CritHEvalCodeStrategy,
    CritMBPPCodeStrategy,
)
from agential.cog.strategies.critic.math_strategy import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
)
from agential.cog.strategies.critic.qa_strategy import (
    CritAmbigNQStrategy,
    CritFEVERStrategy,
    CritHotQAStrategy,
    CritTriviaQAStrategy,
)


class CriticStrategyFactory:
    """A factory class for creating instances of different CRITIC strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> CriticBaseStrategy:
            Returns an instance of the appropriate Critic strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(mode: Dict[str, str], **strategy_kwargs) -> CriticBaseStrategy:
        """Returns an instance of the appropriate Critic strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs: Additional keyword arguments to pass to the strategy's constructor.

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
            if mode["code"] == "mbpp":
                return CritMBPPCodeStrategy(**strategy_kwargs)
            elif mode["code"] == "humaneval":
                return CritHEvalCodeStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
