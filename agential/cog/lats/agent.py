"""LATS agent.

Original Paper: https://arxiv.org/pdf/2310.04406
Paper Repository: https://github.com/lapisrocks/LanguageAgentTreeSearch
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.lats.factory import LATSFactory


class LATSAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        benchmark: str,
        **strategy_kwargs: Any,
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.benchmark = benchmark

        self.strategy = LATSFactory().get_strategy(
            benchmark=self.benchmark,
            llm=self.llm,
            **strategy_kwargs,
        )

    def generate(self):
        pass
    
    def reset(self) -> Any:
        pass