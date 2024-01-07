"""Reflexion Agent implementation.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories: 
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""
from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.messages.human import (
    HumanMessage
)
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.reflexion import (
    _parse_action,
    _format_last_attempt,
    _format_reflections
)
from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import ReflexionReflector
from discussion_agents.cog.eval.reflexion import EM
from discussion_agents.cog.prompts.reflexion import (
    COT,
    COT_REFLECT,
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    cot_reflect_agent_prompt,
)

class ReflexionCoTAgent(BaseAgent):
    self_reflect_llm: BaseChatModel
    action_llm: BaseChatModel
    memory: Optional[ReflexionMemory] = None
    reflector: Optional[ReflexionReflector] = None

    question: str
    context: str
    key: str
    agent_prompt: PromptTemplate = cot_reflect_agent_prompt
    cot_examples: str = COT
    reflect_examples: str = COT_REFLECT

    step_n: int = 0
    answer: str = ""

    finished: bool = False

    def step(self) -> None:
        # Think.
        self.memory.add_memories("\nThought:")
        self.memory.add_memories(" " + self.prompt_agent())
        print(self.memory.load_memories()["scratchpad"].split('\n')[-1])

        # Act.
        action = self.prompt_agent()
        action_type, argument = _parse_action(action)
        self.memory.add_memories("\nAction:")
        self.memory.add_memories(" " + action)
        print(self.memory.load_memories()["scratchpad"].split('\n')[-1])  

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

    def run(self, reflexion_strategy: str = None) -> None:
        if self.step_n > 0 and not self.is_correct() and reflexion_strategy:
            self.reflect(reflexion_strategy)
        self.reset()
        self.step()
        self.step_n += 1

    def reflect(self, strategy: str) -> str:
        reflections = self.reflector.reflect(
            strategy=strategy, 
            examples=self.reflect_examples,
            context=self.context,
            question=self.question,
            scratchpad=self.memory.load_memories()["scratchpad"]
        )

        if strategy == "last_attempt":
            reflections_str = _format_last_attempt(self.question, self.memory.load_memories()["scratchpad"])
        elif strategy == "reflexion":
            reflections_str = _format_reflections(reflections) 
        elif strategy == "last_attempt_and_reflexion":
            reflections_str = _format_last_attempt(self.question, self.memory.load_memories()["scratchpad"])
            reflections_str += "\n" + _format_reflections(reflections, REFLECTION_AFTER_LAST_TRIAL_HEADER)

        return reflections_str

    def reset(self) -> None:
        self.memory.clear()
        self.finished = False

    def prompt_agent(self) -> str:
        prompt = self.agent_prompt.format(
            examples=self.cot_examples,
            reflections=self.reflector.reflections_str,
            context=self.context,
            question=self.question,
            scratchpad=self.memory.load_memories()["scratchpad"]
        )

        out = self.action_llm(
            [
                HumanMessage(
                    content=prompt,
                )
            ]
        ).content
        return format_step(out)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
        
