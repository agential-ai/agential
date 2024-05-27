"""Self-Refine Agent strategies for Math."""

from typing import Dict, Tuple, Any

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.strategies.self_refine.base import SelfRefineBaseStrategy
from agential.cog.functional.self_refine import (
    _prompt_agent,
    _prompt_feedback,
    _prompt_refine,
)


class SelfRefineMathStrategy(SelfRefineBaseStrategy):
    """A strategy class for Math benchmarks using the Self-Refine agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts. Defaults to 2.
    """
    def __init__(self, llm: BaseChatModel, patience: int = 2) -> None:
        """Initialization."""
        self.llm = llm
        self.patience = patience
        self._answer_history = []
        self._prev_code_answer = None
        self.patience_counter = 0
        self._halt = False

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The math question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            str: The generated answer.
        """
        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
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
    ) -> Tuple[str, Dict[str, Any]]:
        critique = _prompt_feedback(
            llm=self.llm,
            question=question,
            examples=examples,
            solution=answer,
            additional_keys=additional_keys,
            prompt=prompt,
        )

        return critique

    def create_output_dict(
        self, answer: str, critique: str
    ) -> Dict[str, str]:
        return {"code": answer, "critique": critique}

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        new_answer = _prompt_refine(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            feedback=critique,
            additional_keys=additional_keys,
            prompt=prompt,
        )
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip()

        return new_answer

    def halting_condition(self) -> bool:
        return self._halt
    
    def reset(self) -> None:
        """Resets the strategy to its initial state.

        Resets internal variables keeping track of halting and answer history.

        Returns:
            bool: True if the reset was successful, False otherwise.
        """
        self._answer_history = []
        self._prev_code_answer = None
        self.patience_counter = 0
        self._halt = False


class SelfRefineGSM8KStrategy(SelfRefineMathStrategy):
    """A strategy class for the GSM8K benchmark using the Self-Refine agent."""

    pass
