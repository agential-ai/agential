"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel

class CriticAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel
    ) -> None:
        super().__init__()

        self.llm = llm

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
