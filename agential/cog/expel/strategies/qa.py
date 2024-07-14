"""ExpeL Agent strategies for QA."""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.reflexion.agent import ReflexionReActAgent
from agential.cog.expel.strategies.base import ExpeLBaseStrategy
from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)

class ExpeLQAStrategy(ExpeLBaseStrategy):
    def __init__(
        self, 
        llm: BaseChatModel,
        reflexion_react_agent: ReflexionReActAgent,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
    ) -> None:
        super().__init__(llm)

        self.llm = llm
        self.reflexion_react_agent = reflexion_react_agent
        self.success_batch_size = success_batch_size

        if not insight_memory:
            self.insight_memory = ExpeLInsightMemory()
        else:
            self.insight_memory = insight_memory

        if not experience_memory:
            self.experience_memory = ExpeLExperienceMemory()
        else:
            self.experience_memory = experience_memory
            self.extract_insights(self.experience_memory.experiences)

    def generate(self) -> str:
        pass

    def get_dynamic_examples(self):
        # Dynamically load in relevant past successful trajectories as fewshot examples.
        dynamic_examples = self.experience_memory.load_memories(
            query=question,
            k_docs=k_docs,
            num_fewshots=num_fewshots,
            max_fewshot_tokens=max_fewshot_tokens,
            reranker_strategy=reranker_strategy,
        )["fewshots"]
        examples = (
            dynamic_examples if dynamic_examples else [examples]  # type: ignore
        )
        examples = "\n\n".join(examples + [END_OF_EXAMPLES_DELIMITER]) + "\n"  # type: ignore

        # Dynamically load in all insights.
        examples += RULE_PREFIX
        insights = self.insight_memory.load_memories()["insights"]
        insights = "".join(
            [f"{i}. {insight['insight']}\n" for i, insight in enumerate(insights)]
        )
        examples += insights

    def gather_experience(self):
        pass

    def extract_insights(self):
        pass

    def update_insights(self):
        pass

