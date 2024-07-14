"""ExpeL Agent strategies for QA."""

from typing import Optional, Dict, Any, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.expel.memory import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)
from agential.cog.expel.strategies.base import ExpeLBaseStrategy
from agential.cog.reflexion.agent import ReflexionReActAgent


class ExpeLQAStrategy(ExpeLBaseStrategy):
    def __init__(
        self,
        llm: BaseChatModel,
        reflexion_react_agent: ReflexionReActAgent,
        experience_memory: Optional[ExpeLExperienceMemory] = None,
        insight_memory: Optional[ExpeLInsightMemory] = None,
        success_batch_size: int = 8,
    ) -> None:
        super().__init__(llm, reflexion_react_agent, experience_memory, insight_memory, success_batch_size)

    def generate(self) -> str:
        pass

    def get_dynamic_examples(
        self,
        question: str,
        examples: str,
        k_docs: int,
        num_fewshots: int,
        max_fewshot_tokens: int,
        reranker_strategy: str,
        additional_keys: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str]]:
        additional_keys = additional_keys.copy()

        # Dynamically load in relevant past successful trajectories as fewshot examples.
        dynamic_examples = self.experience_memory.load_memories(
            query=question,
            k_docs=k_docs,
            num_fewshots=num_fewshots,
            max_fewshot_tokens=max_fewshot_tokens,
            reranker_strategy=reranker_strategy,
        )["fewshots"]
        examples = "\n\n---\n\n".join(
            dynamic_examples if dynamic_examples else [examples]  # type: ignore
        )

        # Dynamically load in all insights.
        insights = self.insight_memory.load_memories()["insights"]
        insights = "".join(
            [f"{i}. {insight['insight']}\n" for i, insight in enumerate(insights)]
        )
        additional_keys.update({"insights": insights})

        return examples, additional_keys

    def gather_experience(self):
        pass

    def extract_insights(self):
        pass

    def update_insights(self):
        pass
