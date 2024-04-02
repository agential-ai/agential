"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.self_refine import (
    _is_halted,
    _prompt_agent,
    _prompt_feedback,
    _prompt_refine,
)
from discussion_agents.cog.modules.memory.self_refine import SelfRefineMemory
from discussion_agents.cog.prompts.self_refine import (
    GSM8K_FEEDBACK_FEWSHOT_EXAMPLES,
    GSM8K_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_FEEDBACK_EXAMPLE_FORMAT_GSM8K,
    SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)


class SelfRefineAgent(BaseAgent):
    def __init__(
        self, llm: BaseChatModel, memory: Optional[SelfRefineMemory] = None
    ) -> None:
        super().__init__()

        self.llm = llm

        if not memory:
            self.memory = SelfRefineMemory()
        else:
            self.memory = memory

    def generate(
        self,
        question: str,
        examples: str = GSM8K_FEWSHOT_EXAMPLES,
        prompt: str = SELF_REFINE_INSTRUCTION_GSM8K,
        feedback_examples: str = GSM8K_FEEDBACK_FEWSHOT_EXAMPLES,
        feedback_prompt: str = SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K,
        refine_examples: str = GSM8K_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt: str = SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        max_attempts: int = 3,
    ) -> str:
        step_n = 0
        while step_n < max_attempts:
            if not step_n:
                solution = _prompt_agent(
                    llm=self.llm, question=question, examples=examples, prompt=prompt
                )

            feedback = _prompt_feedback(
                llm=self.llm,
                examples=feedback_examples,
                solution=solution,
                prompt=feedback_prompt,
            )

            # Halt condition.
            if _is_halted(feedback):
                break
            else:
                improved_solution = _prompt_refine(
                    llm=self.llm,
                    examples=refine_examples,
                    solution=solution,
                    feedback=feedback,
                    prompt=refine_prompt,
                )

                # Continuously update solution & feedback examples.
                feedback_examples = PromptTemplate.from_template(
                    SELF_REFINE_FEEDBACK_EXAMPLE_FORMAT_GSM8K
                ).format(
                    examples=feedback_examples,
                    solution=solution,
                    feedback=feedback,
                    improved_solution=improved_solution,
                )
                solution = improved_solution

            step_n += 1

        return solution

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        pass
