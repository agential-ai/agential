"""Reflexion Agent implementation.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories: 
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""
from typing import Any, Dict, Optional
import tiktoken
from tiktoken import Encoding

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.eval.reflexion import EM
from discussion_agents.cog.functional.reflexion import (
    _prompt_cot_agent,
)
from discussion_agents.cog.modules.memory.react import ReActMemory
from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import ReflexionReflector
from discussion_agents.cog.prompts.reflexion import (
    REFLEXION_COT_FEWSHOT_EXAMPLES,
    REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
    REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES,
    REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES_NO_CONTEXT,
)
from discussion_agents.utils.parse import parse_action


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
    """

    def __init__(
        self,
        self_reflect_llm: BaseChatModel,
        action_llm: BaseChatModel,
        memory: Optional[ReflexionMemory] = None,
        reflector: Optional[ReflexionReflector] = None,
    ) -> None:
        """Initialization with default or provided values."""
        super().__init__()

        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm

        if not memory:
            self.memory = ReflexionMemory()
        else:
            self.memory = memory

        if not reflector and self_reflect_llm:
            self.reflector = ReflexionReflector(llm=self_reflect_llm)
        else:
            self.reflector = reflector

        self._step_n = 0
        self._answer = ""
        self._finished = False

    def generate(
        self,
        question: str,
        key: str,
        context: Optional[str] = None,
        strategy: str = None,
    ) -> str:
        """Generates a response based on the provided context, question, and key.

        The `generate` method internally calls reflect (if possible), resets the memory,
        and generates a thought, action, and the observation (Finish).

        Args:
            question (str): The question to answer.
            key (str): The key to evaluate the correctness of the answer.
            context (Optional[str]): The context or background information. Defaults to None.
            strategy (str, optional): The strategy to use for reflection. Defaults to None.

        Returns:
            out (str): A string output from the ReflexionCoTAgent.
        """
        # Reflect if possible.
        if self._step_n > 0 and not EM(self._answer, key) and strategy:
            self.reflect(strategy, question, context)

        # Reset.
        self.reset()

        out = ""

        # Think.
        self.memory.add_memories("\nThought:")
        thought = _prompt_cot_agent(
            llm=self.action_llm,
            examples=REFLEXION_COT_FEWSHOT_EXAMPLES
            if context
            else REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
            reflections=self.reflector.reflections_str,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            context=context,
        )
        self.memory.add_memories(" " + thought)
        out += self.memory.load_memories()["scratchpad"].split("\n")[-1] + "\n"

        # Act.
        self.memory.add_memories("\nAction:")
        action = _prompt_cot_agent(
            llm=self.action_llm,
            examples=REFLEXION_COT_FEWSHOT_EXAMPLES
            if context
            else REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
            reflections=self.reflector.reflections_str,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            context=context,
        )
        action_type, argument = parse_action(action.strip())
        self.memory.add_memories(" " + action)
        out += self.memory.load_memories()["scratchpad"].split("\n")[-1] + "\n"

        # Observe.
        self.memory.add_memories("\nObservation: ")
        if action_type.lower() == "finish":
            self._answer = argument
            if EM(self._answer, key):
                correctness_str = "Answer is CORRECT"
            else:
                correctness_str = "Answer is INCORRECT"
            self.memory.add_memories(correctness_str)
            out += "\n" + correctness_str
            self._finished = True
        else:
            invalid_action_str = "Invalid action type, please try again."
            self.memory.add_memories(invalid_action_str)
            out += "\n" + invalid_action_str

        self._step_n += 1

        return out

    def reflect(
        self, strategy: str, question: str, context: Optional[str] = None
    ) -> str:
        """Reflects on the previous steps to improve the response.

        Given the agent can reflect (strategy is not `None`), the strategy
        can be of 3 types:
        - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
        - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
        - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
          It first formats the last attempt using 'question' and 'scratchpad', then adds a new reflexion using all the parameters.

        Args:
            strategy (str): The strategy to use for reflection.
            question (str): The question to answer.
            context (Optional[str]): The context or background information. Defaults to None.

        Returns:
            str: Generated reflections based on the strategy.
        """
        _, reflections_str = self.reflector.reflect(
            strategy=strategy,
            examples=REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES
            if context
            else REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES_NO_CONTEXT,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            context=context,
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
        self._finished = False

    def is_finished(self) -> bool:
        """Checks if the agent has finished generating.

        Returns:
            bool: True if the agent has finished, False otherwise.
        """
        return self._finished


class ReflexionReActAgent(BaseAgent):

    def __init__(
        self,
        self_reflect_llm: BaseChatModel,
        action_llm: BaseChatModel,
        memory: Optional[ReActMemory] = None,
        reflector: Optional[ReflexionReflector] = None,
        max_steps: int = 6,
        max_tokens: int = 3896,
        docstore: Optional[DocstoreExplorer] = DocstoreExplorer(Wikipedia()),
        enc: Optional[Encoding] = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__()
