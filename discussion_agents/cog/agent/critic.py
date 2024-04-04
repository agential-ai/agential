"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.functional.critic import _prompt_agent
from discussion_agents.cog.prompts.critic import HOTPOTQA_FEWSHOT_EXAMPLES_COT, CRITIC_INSTRUCTION_HOTPOTQA

class CriticAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel
    ) -> None:
        super().__init__()

        self.llm = llm

    def generate(
        self, 
        question: str,
        examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        prompt: str = CRITIC_INSTRUCTION_HOTPOTQA
    ) -> Any:
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt
        )

