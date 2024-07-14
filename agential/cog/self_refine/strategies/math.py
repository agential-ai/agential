"""Self-Refine Agent strategies for Math."""

from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.self_refine.functional import (
    _prompt_agent,
    _prompt_critique,
    _prompt_refine,
)
from agential.cog.self_refine.strategies.base import SelfRefineBaseStrategy
from agential.eval.em import EM


class SelfRefineMathStrategy(SelfRefineBaseStrategy):
    """A strategy class for Math benchmarks using the Self-Refine agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts. Defaults to 1.
    """

    def __init__(self, llm: BaseChatModel, patience: int = 1) -> None:
        """Initialization."""
        super().__init__(llm, patience)

        self._prev_code_answer = ""
        self.patience_counter = 0
        self._halt = False

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The math question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            str: The generated answer.
        """
        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        answer = answer.split("```python")[-1].split("```")[0].strip()

        return answer

    def generate_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Generates a critique for the provided answer using the given prompt and examples.

        Stops early if patience is reached and answer remains the same.

        Args:
            question (str): The math question that was answered.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            str: The generated critique. If the same incorrect answer is repeated for the number of
                 interactions specified by patience, the halting condition is triggered.
        """
        critique = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        if EM(answer.strip(), self._prev_code_answer, normalize=False):
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                self._halt = True
        else:
            self._prev_code_answer = answer.strip()

        return critique

    def create_output_dict(self, answer: str, critique: str) -> Dict[str, str]:
        """Creates an output dictionary containing the answer and critique.

        Args:
            answer (str): The generated answer.
            critique (str): The generated critique.

        Returns:
            Dict[str, str]: The output dictionary.
        """
        return {"answer": answer, "critique": critique}

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.

        Returns:
            str: The updated answer.
        """
        new_answer = _prompt_refine(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=critique,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip()

        return new_answer

    def halting_condition(self) -> bool:
        """Checks if the halting condition has been met.

        Returns True if the Self-Refine Agent's generated answer remains the same for `patience` number of steps.

        Returns:
            bool: True if the halting condition has been met, False otherwise.
        """
        return self._halt

    def reset(self, **kwargs: Dict[str, Any]) -> None:
        """Resets the strategy to its initial state.

        Resets internal variables keeping track of halting.

        Args:
            **kwargs (Dict[str, Any]): Additional arguments.
        """
        self._prev_code_answer = ""
        self.patience_counter = 0
        self._halt = False


class SelfRefineGSM8KStrategy(SelfRefineMathStrategy):
    """A strategy class for the GSM8K benchmark using the Self-Refine agent."""

    pass


class SelfRefineSVAMPStrategy(SelfRefineMathStrategy):
    """A strategy class for the SVAMP benchmark using the Self-Refine agent."""

    pass


class SelfRefineTabMWPStrategy(SelfRefineMathStrategy):
    """A strategy class for the TABMWP benchmark using the Self-Refine agent."""

    pass
