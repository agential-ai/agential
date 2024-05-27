"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.functional.self_refine import (
    _is_halted,
    _prompt_agent,
    _prompt_feedback,
    _prompt_refine,
)
from agential.cog.modules.memory.self_refine import SelfRefineMemory
from agential.cog.prompts.self_refine import (
    GSM8K_FEEDBACK_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)
from agential.cog.prompts.benchmarks.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_POT,
)


class SelfRefineAgent(BaseAgent):
    """The Self-Refine agent that utilizes the self-refinement process to iteratively improve solutions based on feedback.

    The agent prompts a language model to generate solutions to a given problem, obtains feedback on the generated
    solutions, and then refines the solutions based on this feedback. This process can be repeated a specified number
    of times or until the feedback indicates that no further improvements are needed.

    Attributes:
        llm (BaseChatModel): The language model used for generating solutions, feedback, and refinements.
        memory (SelfRefineMemory): A memory module for storing solutions and feedback across refinement iterations.
    """

    def __init__(
        self, llm: BaseChatModel, mode: Dict[str, str],
        **strategy_kwargs
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm

        if not memory:
            self.memory = SelfRefineMemory()
        else:
            self.memory = memory

    def generate(
        self,
        question: str,
        examples: str = GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt: str = SELF_REFINE_INSTRUCTION_GSM8K,
        feedback_examples: str = GSM8K_FEEDBACK_FEWSHOT_EXAMPLES,
        feedback_prompt: str = SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K,
        refine_examples: str = GSM8K_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt: str = SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        max_attempts: int = 3,
        reset: bool = True,
    ) -> List[str]:
        """Generates a refined solution for a given question through an iterative self-refinement process.

        The process includes generating initial solutions, soliciting feedback, and refining the solution
        based on feedback, repeated for a maximum number of attempts or until feedback indicates satisfaction.

        Args:
            question (str): The question or problem to solve.
            examples (str): Precedent examples to guide initial solution generation. Defaults to GSM8K_FEWSHOT_EXAMPLES_POT.
            prompt (str): Instructional prompt for initial solution generation. Defaults to SELF_REFINE_INSTRUCTION_GSM8K.
            feedback_examples (str): Precedent examples to guide feedback generation. Defaults to GSM8K_FEEDBACK_FEWSHOT_EXAMPLES.
            feedback_prompt (str): Instructional prompt for feedback generation. Defaults to SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K.
            refine_examples (str): Precedent examples to guide solution refinement. Defaults to GSM8K_REFINE_FEWSHOT_EXAMPLES.
            refine_prompt (str): Instructional prompt for refining the solution. Defaults to SELF_REFINE_REFINE_INSTRUCTION_GSM8K.
            max_attempts (int): Maximum number of refinement iterations.
            reset (bool): Resets the agent's state. Defaults to True.

        Returns:
            str: The final refined solution.
        """
        if reset:
            self.reset()

        solution = _prompt_agent(
            llm=self.llm, question=question, examples=examples, prompt=prompt
        )

        step_n = 0
        while step_n < max_attempts:
            # Generate feedback.
            feedback = _prompt_feedback(
                llm=self.llm,
                question=question,
                examples=feedback_examples,
                solution=solution,
                prompt=feedback_prompt,
            )

            if _is_halted(feedback):
                break

            solution = _prompt_refine(
                llm=self.llm,
                question=question,
                examples=refine_examples,
                solution=solution,
                feedback=feedback,
                prompt=refine_prompt,
            )

            step_n += 1

        return

        # step_n = 0
        # while step_n < max_attempts:
        #     if not step_n:
        #         solution = _prompt_agent(
        #             llm=self.llm, question=question, examples=examples, prompt=prompt
        #         )

        #     feedback = _prompt_feedback(
        #         llm=self.llm,
        #         examples=feedback_examples,
        #         solution=solution,
        #         prompt=feedback_prompt,
        #     )

        #     # Update memory with solution, feedback pair (tracks all previous solutions and feedbacks).
        #     self.memory.add_memories(solution, feedback)

        #     # Halt condition.
        #     if _is_halted(feedback):
        #         break
        #     else:
        #         improved_solution = _prompt_refine(
        #             llm=self.llm,
        #             examples=refine_examples,
        #             solution=solution,
        #             feedback=feedback,
        #             prompt=refine_prompt,
        #         )

        #         # Continuously update solution & feedback examples.
        #         feedback_examples = PromptTemplate.from_template(
        #             feedback_format
        #         ).format(
        #             examples=feedback_examples,
        #             solution=solution,
        #             feedback=feedback,
        #             improved_solution=improved_solution,
        #         )
        #         solution = improved_solution

        #         # Add the new solution (no feedback) to the memory, if applicable.
        #         self.memory.add_memories(solution, "")

        #     step_n += 1

        # return self.memory.load_memories()["solution"]

    def reset(self) -> None:
        """Resets the agent's memory."""
        self.memory.clear()

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()
