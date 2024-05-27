"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.base import BaseAgent
from agential.cog.functional.self_refine import (
    _prompt_agent,
    _prompt_critique,
    _prompt_refine,
)
from agential.cog.prompts.benchmarks.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_POT,
)
from agential.cog.prompts.self_refine import (
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)
from agential.cog.strategies.strategy_factory import SelfRefineStrategyFactory


class SelfRefineAgent(BaseAgent):
    """The Self-Refine agent that utilizes the self-refinement process to iteratively improve solutions based on critique.

    The agent prompts a language model to generate solutions to a given problem, obtains critique on the generated
    solutions, and then refines the solutions based on this critique. This process can be repeated a specified number
    of times or until the critique indicates that no further improvements are needed.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        mode (Dict[str, str]): A dictionary specifying the CRITIC agent's mode and the benchmark.
            For example, {"qa": "hotpotqa"}, {"math": "gsm8k"}, or {"code": "mbpp"}.
    """

    def __init__(
        self, llm: BaseChatModel, mode: Dict[str, str], **strategy_kwargs
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.mode = mode

        self.strategy = SelfRefineStrategyFactory().get_strategy(
            mode=self.mode, llm=self.llm, **strategy_kwargs
        )

    def generate(
        self,
        question: str,
        examples: str = GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt: str = SELF_REFINE_INSTRUCTION_GSM8K,
        critique_examples: str = GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        critique_prompt: str = SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        refine_examples: str = GSM8K_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt: str = SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        additional_keys: str = {},
        critique_additional_keys: Dict[str, str] = {},
        refine_additional_keys: Dict[str, str] = {},
        max_interactions: int = 3,
        reset: bool = True,
    ) -> List[str]:
        """Generates a refined solution for a given question through an iterative self-refinement process.

        The process includes generating initial solutions, soliciting critique, and refining the solution
        based on critique, repeated for a maximum number of attempts or until critique indicates satisfaction.

        Args:
            question (str): The question or problem to solve.
            examples (str): Precedent examples to guide initial solution generation. Defaults to GSM8K_FEWSHOT_EXAMPLES_POT.
            prompt (str): Instructional prompt for initial solution generation. Defaults to SELF_REFINE_INSTRUCTION_GSM8K.
            critique_examples (str): Precedent examples to guide critique generation. Defaults to GSM8K_CRITIQUE_FEWSHOT_EXAMPLES.
            critique_prompt (str): Instructional prompt for critique generation. Defaults to SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K.
            refine_examples (str): Precedent examples to guide solution refinement. Defaults to GSM8K_REFINE_FEWSHOT_EXAMPLES.
            refine_prompt (str): Instructional prompt for refining the solution. Defaults to SELF_REFINE_REFINE_INSTRUCTION_GSM8K.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt. Defaults to {}.
            refine_additional_keys (Dict[str, str]): Additional keys to format the refine_prompt. Defaults to {}.
            max_interactions (int): Maximum number of refinement iterations.
            reset (bool): Resets the agent's state. Defaults to True.

        Returns:
            str: The final refined solution.
        """
        if reset:
            self.reset()

        out = []

        # Initial answer generation.
        answer = self.strategy.generate(question, examples, prompt, additional_keys)

        for _ in range(max_interactions):
            critique = self.strategy.generate_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
            )

            out.append(self.strategy.create_output_dict(answer, critique))

            if self.strategy.halting_condition():
                break

            answer = self.strategy.update_answer_based_on_critique(
                question=question,
                examples=refine_examples,
                answer=answer,
                critique=critique,
                prompt=refine_prompt,
                additional_keys=refine_additional_keys,
            )

        return out

    def reset(self) -> None:
        self.strategy.reset()
