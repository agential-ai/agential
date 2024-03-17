from typing import Optional

from discussion_agents.cog.modules.memory.expel import ExpeLExperienceMemory, ExpeLInsightMemory
from discussion_agents.cog.agent.base import BaseAgent


class ExpeLAgent(BaseAgent):
    def __init__(
        self,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None
    ) -> None:
        super().__init__()

        if not experience_memory:
            self.experience_memory = ExpeLExperienceMemory()
        else:
            self.experience_memory = experience_memory

        if not insight_memory:
            self.insight_memory = ExpeLInsightMemory()
        else:
            self.insight_memory = insight_memory

    def generate(self, question: str, key: str, reset: bool = True):
        pass