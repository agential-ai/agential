"""Self-Refine Agent.

Original Webpage: https://selfrefine.info/
Paper Repository: https://github.com/madaan/self-refine
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.base.agent import BaseAgent
from agential.cog.self_refine.factory import (
    SELF_REFINE_BENCHMARK_FEWSHOTS,
    SelfRefineFactory,
)
from agential.cog.self_refine.output import SelfRefineOutput


class SelfRefineAgent(BaseAgent):
    """The Self-Refine agent that utilizes the self-refinement process to iteratively improve solutions based on critique.

    The agent prompts a language model to generate solutions to a given problem, obtains critique on the generated
    solutions, and then refines the solutions based on this critique. This process can be repeated a specified number
    of times or until the critique indicates that no further improvements are needed.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        benchmark (str): The benchmark name.
        **strategy_kwargs (Dict[str, Any]): Additional strategy-specific arguments.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        benchmark: str,
        **strategy_kwargs: Dict[str, Any],
    ) -> None:
        """Initialization."""
        super().__init__()

        self.llm = llm
        self.benchmark = benchmark

        self.strategy = SelfRefineFactory().get_strategy(
            benchmark=self.benchmark, llm=self.llm, **strategy_kwargs
        )

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        critique_examples: str = "",
        critique_prompt: str = "",
        refine_examples: str = "",
        refine_prompt: str = "",
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        refine_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        max_interactions: int = 3,
        reset: bool = True,
    ) -> List[SelfRefineOutput]:
        """Generates a refined solution for a given question through an iterative self-refinement process.

        The process includes generating initial solutions, soliciting critique, and refining the solution
        based on critique, repeated for a maximum number of attempts or until critique indicates satisfaction.

        Args:
            question (str): The question or problem to solve.
            examples (str, optional): Precedent examples to guide initial solution generation. Defaults to "".
            prompt (str, optional): Instructional prompt for initial solution generation. Defaults to "".
            critique_examples (str, optional): Precedent examples to guide critique generation. Defaults to "".
            critique_prompt (str, optional): Instructional prompt for critique generation. Defaults to "".
            refine_examples (str, optional): Precedent examples to guide solution refinement. Defaults to "".
            refine_prompt (str, optional): Instructional prompt for refining the solution. Defaults to "".
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt. Defaults to {}.
            refine_additional_keys (Dict[str, str]): Additional keys to format the refine_prompt. Defaults to {}.
            fewshot_type (str): The type of few-shot examples to use. Defaults to "".
            max_interactions (int): Maximum number of refinement iterations.
            reset (bool): Resets the agent's state. Defaults to True.

        Returns:
            List[SelfRefineOutput]: A list of answers and critiques.
        """
        if (
            not prompt
            or not critique_prompt
            or not examples
            or not critique_examples
            or not refine_examples
            or not refine_prompt
        ):
            if not fewshot_type:
                fewshot_type = SELF_REFINE_BENCHMARK_FEWSHOTS[self.benchmark][0]  # type: ignore
            fewshots = SelfRefineFactory.get_fewshots(
                benchmark=self.benchmark, fewshot_type=fewshot_type
            )
            prompts = SelfRefineFactory.get_prompts(benchmark=self.benchmark)
            examples = fewshots["examples"]
            critique_examples = fewshots["critique_examples"]
            refine_examples = fewshots["refine_examples"]
            prompt = prompts["prompt"]
            critique_prompt = prompts["critique_prompt"]
            refine_prompt = prompts["refine_prompt"]

        if reset:
            self.reset()

        out = []

        # Initial answer generation.
        answer = self.strategy.generate(question, examples, prompt, additional_keys)

        for _ in range(max_interactions):
            # Generate critique.
            critique = self.strategy.generate_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
            )

            out.append(
                SelfRefineOutput(**self.strategy.create_output_dict(answer, critique))
            )

            if self.strategy.halting_condition():
                break

            # Improve answer based on critique.
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
        """Resets the agent's internal state."""
        self.strategy.reset()
