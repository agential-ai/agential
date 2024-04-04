"""CRITIC Agent.

GitHub Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
Original Paper: http://arxiv.org/abs/2305.11738
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.functional.critic import _prompt_agent, _prompt_critique, _build_critique_format_prompt
from discussion_agents.cog.prompts.critic import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT, 
    CRITIC_INSTRUCTION_HOTPOTQA,
    HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_FORMAT_HOTPOTQA
)

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
        prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
        critique_examples: str = HOTPOTQA_FEWSHOT_EXAMPLES_CRITIC,
        critique_prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
        critique_format_prompt: str = CRITIC_CRITIQUE_FORMAT_HOTPOTQA,
        max_interactions: int = 7,
        use_tool: bool = True
    ) -> Any:
        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt
        )

        exist_query = []
        exist_evidence = set()
        for idx in range(max_interactions):
            critique = _prompt_critique(
                llm=self.llm,
                question=question,
                examples=critique_examples,
                answer=answer,
                prompt=critique_prompt
            )

            if "> Search Query: " in critique:
                _, search_query = critique.split("> Search Query:")[:2]
                search_query = search_query.split("\n")[0].strip()
                a = _build_critique_format_prompt(
                    question=question,
                    examples=critique_examples,
                    answer=answer,
                    critique=critique,
                    prompt=critique_format_prompt
                )

                if use_tool:
                    exist_query.append(search_query)
                    
                else:
                    pass



