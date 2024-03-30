"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.functional.self_refine import _prompt_agent
from discussion_agents.cog.prompts.self_refine import GSM8K_FEWSHOT_EXAMPLES, SELF_REFINE_INSTRUCTION_GSM8K

class SelfRefineAgent(BaseAgent):
    def __init__(
        self, 
        llm: BaseChatModel
    ) -> None:
        super().__init__()

        self.llm = llm

    def generate(
        self, 
        question: str,
        examples: str = GSM8K_FEWSHOT_EXAMPLES,
        prompt: str = SELF_REFINE_INSTRUCTION_GSM8K,
        max_attempts: int = 3,
        question_prefix="# Q: ",
        answer_prefix="# solution using Python:",
        intra_example_sep="\n"
    ) -> Any:

        step_n = 0
        while step_n < max_attempts:

            if not step_n:
                out = _prompt_agent(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    question_prefix=question_prefix,
                    intra_example_sep=intra_example_sep,
                    answer_prefix=answer_prefix,
                    prompt=prompt
                )

            fb_and_maybe_soln = task_feedback(solution=out)
            
            if "it is correct" in fb_and_maybe_soln["feedback"].lower():
                break

            out = fb_and_maybe_soln["solution"]

            step_n += 1

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        pass