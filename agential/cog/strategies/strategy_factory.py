from agential.cog.strategies.critic.qa_strategy import QAStrategy
from agential.cog.strategies.critic.math_strategy import MathStrategy


class CriticStrategyFactory:
    @staticmethod
    def get_strategy(mode: str, **strategy_kwargs):
        if mode == "qa":
            return QAStrategy(**strategy_kwargs)
        elif mode == "math":
            return MathStrategy(**strategy_kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
