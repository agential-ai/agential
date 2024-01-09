"""Reflexion Agent implementation.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories: 
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""
from typing import Optional, Dict, Any
from pydantic import root_validator

from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.reflexion import (
    _parse_action,
)
from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import ReflexionReflector
from discussion_agents.cog.eval.reflexion import EM
from discussion_agents.cog.functional.reflexion import _prompt_cot_agent
from discussion_agents.cog.prompts.reflexion import (
    COT,
    COT_REFLECT,
)

class ReflexionCoTAgent(BaseAgent):
    self_reflect_llm: BaseChatModel
    action_llm: BaseChatModel
    memory: Optional[ReflexionMemory] = None
    reflector: Optional[ReflexionReflector] = None

    @root_validator(pre=False)
    def set_args(cls: Any, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set default arguments."""
        self_reflect_llm = values.get("self_reflect_llm")
        memory = values.get("memory")
        reflector = values.get("reflector")

        if not memory:
            values["memory"] = ReflexionMemory()
        if not reflector:
            values["reflector"] = ReflexionReflector(llm=self_reflect_llm) 
        return values

    step_n: int = 0
    answer: str = ""
    finished: bool = False

    def generate(
        self, 
        context: str, 
        question: str, 
        key: str, 
        reflexion_strategy: str = None
    ) -> None:
        # Reflect if possible.
        if self.step_n > 0 and not self.is_correct(self.answer, key) and reflexion_strategy:
            self.reflect(context, question, reflexion_strategy)

        # Reset.
        self.reset()

        # Think.
        self.memory.add_memories("\nThought:")
        self.memory.add_memories(" " + \
            _prompt_cot_agent(
                llm=self.action_llm,
                examples=COT,
                reflections=self.reflector.reflections_str,
                context=context,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"]
            )
        )
        print(self.memory.load_memories()["scratchpad"].split('\n')[-1])

        # Act.
        action = _prompt_cot_agent(
            llm=self.action_llm,
            examples=COT,
            reflections=self.reflector.reflections_str,
            context=context,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"]
        )
        action_type, argument = _parse_action(action)
        self.memory.add_memories("\nAction:")
        self.memory.add_memories(" " + action)
        print(self.memory.load_memories()["scratchpad"].split('\n')[-1])  

        # Observe.
        self.memory.add_memories("\nObservation:")
        if action_type == "Finish":
            self.answer = argument
            if self.is_correct():
                self.memory.add_memories("Answer is CORRECT")
            else: 
                self.memory.add_memories("Answer is INCORRECT")
            self.finished = True
        else:
            print('Invalid action type, please try again.')

        self.step_n += 1

    def reflect(self, context: str, question: str, strategy: str) -> str:
        _, reflections_str = self.reflector.reflect(
            strategy=strategy, 
            examples=COT_REFLECT,
            context=context,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"]
        )

        return reflections_str

    def retrieve(self) -> Dict[str, Any]:
        return self.memory.load_memories()

    def reset(self) -> None:
        self.memory.clear()
        self.finished = False

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self, answer: str, key: str) -> bool:
        return EM(answer, key)
        
