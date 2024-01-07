"""Reflecting module for Reflexion."""
from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.modules.reflect.base import BaseReflector
from discussion_agents.cog.functional.reflexion import reflect

class ReflexionReflector(BaseReflector):
    """Reflexion module for reflecting.

    This class encapsulates the logic for reflecting on a given context, question, and scratchpad content using various
    strategies. It leverages a language model to generate reflections and maintains a list of these reflections.

    Attributes:
        llm (BaseChatModel): A language model used for generating reflections.
        reflections (List[str]): A list to store the generated reflections.
    """
    llm: BaseChatModel
    reflections: Optional[List[str]] = []

    def reflect(
        self,
        strategy: str,
        examples: str,
        context: str,
        question: str,
        scratchpad: str
    ) -> List[str]:
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
            List[str]: The updated list of reflections based on the selected strategy.

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
            scratchpad=scratchpad
        )

        self.reflections = reflections

        return reflections