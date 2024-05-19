from agential.cog.strategies.critic.qa_strategy import QAStrategy
from agential.cog.strategies.critic.math_strategy import MathStrategy

class CriticStrategyFactory:
    @staticmethod
    def get_strategy(mode: str):
        if mode == "qa":
            return QAStrategy()
        elif mode == "math":
            return MathStrategy()
        # Add other modes and their strategies as needed
        else:
            raise ValueError(f"Unsupported mode: {mode}")
