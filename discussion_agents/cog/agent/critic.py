"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent

class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return super().generate(*args, **kwargs)
    
    