"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel

class SelfRefineAgent(BaseAgent):
    def __init__(
        self, 
        llm: BaseChatModel
    ) -> None:
        super().__init__()

        self.llm = llm

    def generate(
        self, 
        examples: str,
        prompt: str,
        max_attempts: int = 3
    ) -> Any:

        step_n = 0
        while step_n < max_attempts:

            if not step_n:
                solution = task_init(solution=question)

            fb_and_maybe_soln = task_feedback(solution=solution)
            

            log.append({"attempt": n_attempts, "solution_curr": solution, "solution_fixed": fb_and_maybe_soln["solution"], "feedback": fb_and_maybe_soln["feedback"]})

            if "it is correct" in fb_and_maybe_soln["feedback"].lower():
                break

            solution = fb_and_maybe_soln["solution"]

            step_n += 1

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        pass