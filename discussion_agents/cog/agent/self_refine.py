"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent

class SelfRefineAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return super().generate(*args, **kwargs)
    
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return super().reset(*args, **kwargs)
    
    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        return super().reflect(*args, **kwargs)
    
    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        return super().retrieve(*args, **kwargs)