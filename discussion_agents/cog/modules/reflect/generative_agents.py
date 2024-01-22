"""Reflecting module for Generative Agents."""
from datetime import datetime
from typing import Any, List, Optional, Union

from langchain_core.retrievers import BaseRetriever

from discussion_agents.cog.functional.generative_agents import (
    reflect,
)
from discussion_agents.cog.modules.reflect.base import BaseReflector


class GenerativeAgentReflector(BaseReflector):
    """A reflector class designed for generative agents, specializing in generating insights from observations.

    This class extends BaseReflector and is equipped to analyze and interpret observations,
    providing valuable insights. It utilizes a language model (LLM) and a retrieval system (implemented via BaseRetriever)
    to process and understand the observations.

    Attributes:
        llm (LLM): An instance of a language model, integral to the process of interpreting observations and generating insights.
        retriever (BaseRetriever): A retrieval system that aids in the reflection process, potentially by accessing relevant stored memories or data.

    The class offers a `reflect` method, which takes a set of observations and returns insights based on these observations.
    """

    def __init__(self, llm: Any, retriever: BaseRetriever) -> None:
        """Initialization."""
        super().__init__(llm)
        self.retriever = retriever

    def reflect(
        self, observations: Union[str, List[str]], now: Optional[datetime] = None
    ) -> List[List[str]]:
        """Analyzes observations and generates insights using the language model and retriever.

        This method processes a given set of observations and interprets them to produce insights. It uses the class's language model and retrieval system to understand and contextualize the observations, taking into account the current time if provided.

        Args:
            observations (Union[str, List[str]]): The observations to be reflected upon. This can be a single observation or a list of observations.
            now (Optional[datetime], optional): The current time, used to provide temporal context to the reflection process. Defaults to None.

        Returns:
            List[str]: A list of insights generated from the observations. These insights are strings that represent the model's interpretation and understanding of the input observations.

        The method internally calls the `reflect` function, delegating the process of generating insights.
        """
        _, insights = reflect(
            observations=observations, llm=self.llm, retriever=self.retriever, now=now
        )
        return insights

    def clear(self, retriever: BaseRetriever) -> None:
        """Clears the retriever and sets to specified retriever."""
        self.retriever = retriever
