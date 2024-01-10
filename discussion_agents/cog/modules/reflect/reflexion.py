"""Reflecting module for Reflexion."""
from typing import List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.functional.reflexion import (
    _format_last_attempt,
    _format_reflections,
    reflect,
)
from discussion_agents.cog.modules.reflect.base import BaseReflector
from discussion_agents.cog.prompts.reflexion import (
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
)


class ReflexionReflector(BaseReflector):
    """Reflexion module for reflecting.

    This class encapsulates the logic for reflecting on a given context, question, and scratchpad content using various
    strategies. It leverages a language model to generate reflections and maintains a list of these reflections.

    Attributes:
        llm (BaseChatModel): A language model used for generating reflections.
        reflections (Optional[List[str]]): A list to store the generated reflections.
        reflections_str (Optional[str]): The reflections formatted into a string.
    """

    llm: BaseChatModel
    reflections: Optional[List[str]] = []
    reflections_str: Optional[str] = ""

    def reflect(
        self, strategy: str, examples: str, context: str, question: str, scratchpad: str
    ) -> Tuple[List[str], str]:
        """Wrapper around Reflexion's `reflect` method in functional.

        This method calls the appropriate reflection function based on the provided strategy, passing in the necessary
        parameters including the language model, context, question, and scratchpad. It then updates the internal
        reflections list with the newly generated reflections.

        Args:
            strategy (str): The reflection strategy to be used ('last_attempt', 'reflexion', or 'last_attempt_and_reflexion').
            examples (str): Example inputs for the prompt template.
            context (str): The context of the conversation or query.
            question (str): The question being addressed.
            scratchpad (str): The scratchpad content related to the question.

        Returns:
            Tuple[List[str], str]: A tuple of the updated list of reflections based on the selected strategy and the formatted
                reflections.

        Raises:
            NotImplementedError: If an unknown reflection strategy is specified.
        """
        reflections = reflect(
            strategy=strategy,
            llm=self.llm,
            reflections=self.reflections,
            examples=examples,
            context=context,
            question=question,
            scratchpad=scratchpad,
        )

        if strategy == "last_attempt":
            reflections_str = _format_last_attempt(question, scratchpad)
        elif strategy == "reflexion":
            reflections_str = _format_reflections(reflections)
        elif strategy == "last_attempt_and_reflexion":
            reflections_str = _format_last_attempt(question, scratchpad)
            reflections_str += "\n" + _format_reflections(
                reflections, REFLECTION_AFTER_LAST_TRIAL_HEADER
            )

        self.reflections = reflections
        self.reflections_str = reflections_str

        return reflections, reflections_str
