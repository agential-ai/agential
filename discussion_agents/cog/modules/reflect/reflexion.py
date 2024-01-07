"""Reflecting module for Reflexion."""
from typing import Any, List, Tuple
from langchain_core.language_models.chat_models import BaseChatModel

from discussion_agents.cog.modules.reflect.base import BaseReflector
from discussion_agents.cog.functional.reflexion import reflect

class ReflexionReflector(BaseReflector):
    llm: BaseChatModel
    reflections: List[str] = []

    def reflect(
        self,
        strategy: str,
        examples: str,
        context: str,
        question: str,
        scratchpad: str
    ) -> List[str]:
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