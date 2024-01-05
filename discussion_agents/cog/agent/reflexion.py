"""Reflexion Agent."""
from typing import Any

from discussion_agents.cog.agent.base import BaseAgent

class ReflexionCoT(BaseAgent):
    self_reflect_llm: Any
    action_llm: Any

    question: str,
    context: str,
    key: str,
    agent_prompt: PromptTemplate = cot_reflect_agent_prompt,
    reflect_prompt: PromptTemplate = cot_reflect_prompt,
    cot_examples: str = COT,
    reflect_examples: str = COT_REFLECT

    def run(self, reflexion_strategy: str = None) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1