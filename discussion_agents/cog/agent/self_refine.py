"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any
from discussion_agents.cog.agent.base import BaseAgent
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.functional.self_refine import _prompt_agent, _prompt_feedback
from discussion_agents.cog.prompts.self_refine import (
    GSM8K_FEWSHOT_EXAMPLES, 
    GSM8K_FEEDBACK_FEWSHOT_EXAMPLES,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K,
    GSM8K_FEEDBACK_INSTRUCTION
)

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
        feedback_examples: str = GSM8K_FEEDBACK_FEWSHOT_EXAMPLES,
        feedback_prompt: str = SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K,
        feedback_instruction: str = GSM8K_FEEDBACK_INSTRUCTION,
        max_attempts: int = 3,
        question_prefix="# Q: ",
        answer_prefix="# solution using Python:",
        intra_example_sep="\n",
        inter_example_sep="\n\n### END ###\n\n"
    ) -> str:

        step_n = 0
        while step_n < max_attempts:

            if not step_n:
                solution = _prompt_agent(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    question_prefix=question_prefix,
                    intra_example_sep=intra_example_sep,
                    answer_prefix=answer_prefix,
                    prompt=prompt
                )

            feedback_and_improved_solution = _prompt_feedback(
                llm=self.llm,
                examples=feedback_examples,
                question_prefix="",
                solution=solution,
                intra_example_sep="\n\n",
                feedback_instruction=feedback_instruction,
                answer_prefix="",
                prompt=feedback_prompt
            )

            # Continuously update feedback examples.
            feedback = feedback_and_improved_solution.split("def solution():")[0]
            improved_solution = "def solution():" + \
                feedback_and_improved_solution.split("def solution():")[1].split("### END")[0].rstrip()
            prefix = f"""{question_prefix}{solution}{intra_example_sep}{feedback_instruction}{answer_prefix}"""
            gen_ans = f"""

            {feedback}

            {improved_solution.rstrip()}{inter_example_sep}"""
            new_example = f"{prefix}{gen_ans}"
            feedback_examples = f"{feedback_examples}{new_example}"

            # Continuously update solution.
            solution = improved_solution

            # Halt condition.
            if "it is correct" in feedback.lower():
                break

            step_n += 1

        return solution

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        pass