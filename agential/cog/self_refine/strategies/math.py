"""Self-Refine Agent strategies for Math."""

from typing import Any, Dict, Tuple

from agential.cog.self_refine.functional import (
    _prompt_agent,
    _prompt_critique,
    _prompt_refine,
)
from agential.cog.self_refine.strategies.general import SelfRefineGeneralStrategy
from agential.eval.em import EM
from agential.llm.llm import BaseLLM, Response


class SelfRefineMathStrategy(SelfRefineGeneralStrategy):
    """A strategy class for Math benchmarks using the Self-Refine agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts. Defaults to 1.
    """

    def __init__(self, llm: BaseLLM, patience: int = 1) -> None:
        """Initialization."""
        super().__init__(llm, patience)

        self._prev_answer = ""
        self.patience_counter = 0

    def generate_answer(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            Tuple[str, Response]: The generated answer and the response from the language model.
        """
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        answer = out.output_text
        answer = answer.strip().split("```python")[-1].split("```")[0].strip()

        return answer, out

    def generate_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, bool, Response]:
        """Generates a critique for the provided answer using the given prompt and examples.

        Stops early if patience is reached and answer remains the same.

        Args:
            question (str): The qa question that was answered.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            Tuple[str, bool, Response]: The critique, a boolean indicating it's finished, and the model response.
        """
        out = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        critique = out.output_text.strip()

        finished = False
        if EM(answer.strip(), self._prev_answer, normalize=False):
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                finished = True
        else:
            self._prev_answer = answer.strip()

        return critique, finished, out

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.

        Returns:
            Tuple[str, Response]: The updated answer and the model response.
        """
        out = _prompt_refine(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        new_answer = (
            out.output_text.strip().split("```python")[-1].split("```")[0].strip()
        )

        return new_answer, out

    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction has finished.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        return finished

    def reset(self) -> None:
        """Resets the strategy to its initial state."""
        self._prev_answer = ""
        self.patience_counter = 0


class SelfRefineGSM8KStrategy(SelfRefineMathStrategy):
    """A strategy class for the GSM8K benchmark using the Self-Refine agent."""

    pass


class SelfRefineSVAMPStrategy(SelfRefineMathStrategy):
    """A strategy class for the SVAMP benchmark using the Self-Refine agent."""

    pass


class SelfRefineTabMWPStrategy(SelfRefineMathStrategy):
    """A strategy class for the TABMWP benchmark using the Self-Refine agent."""

    pass
