"""Reflexion Agent implementation.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories: 
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic.v1 import root_validator

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.eval.reflexion import EM
from discussion_agents.cog.functional.reflexion import (
    _parse_action,
    _prompt_cot_agent,
)
from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import ReflexionReflector
from discussion_agents.cog.prompts.reflexion import (
    COT,
    COT_REFLECT,
)


class ReflexionCoTAgent(BaseAgent):
    """Reflexion with Chain-of-Thought actor.

    Attributes:
        self_reflect_llm (BaseChatModel): The language model used for self-reflection.
        action_llm (BaseChatModel): The language model used for generating thoughts/actions.
        memory (Optional[ReflexionMemory]): An optional memory module to store the agent's internal state.
        reflector (Optional[ReflexionReflector]): An optional reflector module for guided self-reflection.

    Methods:
        set_args(cls, values): A class method for setting default arguments.
        generate(context, question, key, strategy): Generates a response based on the given context, question, and strategy.
        reflect(context, question, strategy): Reflects on the previous response and modifies the strategy accordingly.
        retrieve(): Retrieves the current memory state of the agent.
        reset(): Resets the agent's state for a new problem-solving session.
        is_finished(): Checks if the problem-solving process has concluded.
        is_correct(answer, key): Evaluates if the provided answer is correct.
    """

    self_reflect_llm: BaseChatModel
    action_llm: BaseChatModel
    memory: Optional[ReflexionMemory] = None
    reflector: Optional[ReflexionReflector] = None

    @root_validator(pre=False, skip_on_failure=True)
    def set_args(cls: Any, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set default arguments."""
        self_reflect_llm = values.get("self_reflect_llm")
        memory = values.get("memory")
        reflector = values.get("reflector")

        if not memory:
            values["memory"] = ReflexionMemory()
        if self_reflect_llm and not reflector:
            values["reflector"] = ReflexionReflector(llm=self_reflect_llm)
        return values

    step_n: int = 0  #: :meta private:
    answer: str = ""  #: :meta private:
    finished: bool = False  #: :meta private:

    def generate(
        self, context: str, question: str, key: str, strategy: str = None
    ) -> None:
        """Generates a response based on the provided context, question, and key.

        The `generate` method internally calls reflect (if possible), resets the memory,
        and generates a thought, action, and the observation (Finish).

        Args:
            context (str): The context or background information.
            question (str): The question to answer.
            key (str): The key to evaluate the correctness of the answer.
            strategy (str, optional): The strategy to use for reflection. Defaults to None.
        """
        # Reflect if possible.
        if self.step_n > 0 and not self.is_correct(self.answer, key) and strategy:
            self.reflect(context, question, strategy)

        # Reset.
        self.reset()

        # Think.
        self.memory.add_memories("\nThought:")
        self.memory.add_memories(
            " "
            + _prompt_cot_agent(
                llm=self.action_llm,
                examples=COT,
                reflections=self.reflector.reflections_str,
                context=context,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
            )
        )
        print(self.memory.load_memories()["scratchpad"].split("\n")[-1])

        # Act.
        self.memory.add_memories("\nAction:")
        action = _prompt_cot_agent(
            llm=self.action_llm,
            examples=COT,
            reflections=self.reflector.reflections_str,
            context=context,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
        )
        action_type, argument = _parse_action(action)
        self.memory.add_memories(" " + action)
        print(self.memory.load_memories()["scratchpad"].split("\n")[-1])

        # Observe.
        self.memory.add_memories("\nObservation:")
        if action_type == "Finish":
            self.answer = argument
            if self.is_correct(self.answer, key):
                self.memory.add_memories("Answer is CORRECT")
            else:
                self.memory.add_memories("Answer is INCORRECT")
            self.finished = True
        else:
            print("Invalid action type, please try again.")

        self.step_n += 1

    def reflect(self, context: str, question: str, strategy: str) -> str:
        """Reflects on the previous steps to improve the response.

        Given the agent can reflect (strategy is not `None`), the strategy
        can be of 3 types:
        - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
        - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
        - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
          It first formats the last attempt using 'question' and 'scratchpad', then adds a new reflexion using all the parameters.


        Args:
            context (str): The context or background information.
            question (str): The question to answer.
            strategy (str): The strategy to use for reflection.

        Returns:
            str: Generated reflections based on the strategy.
        """
        _, reflections_str = self.reflector.reflect(
            strategy=strategy,
            examples=COT_REFLECT,
            context=context,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
        )

        return reflections_str

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()

    def reset(self) -> None:
        """Resets the agent's memory and state."""
        self.memory.clear()
        self.finished = False

    def is_finished(self) -> bool:
        """Checks if the agent has finished generating.

        Returns:
            bool: True if the agent has finished, False otherwise.
        """
        return self.finished

    def is_correct(self, answer: str, key: str) -> bool:
        """Evaluates the correctness of the generated answer against the key using exact-math grading.

        Args:
            answer (str): The generated answer.
            key (str): The key to evaluate against.

        Returns:
            bool: True if the answer is correct, False otherwise.
        """
        return EM(answer, key)
