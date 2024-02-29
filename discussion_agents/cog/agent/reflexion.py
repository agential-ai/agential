"""Reflexion Agent implementation.

Original Paper: https://arxiv.org/abs/2303.11366
Paper Repositories:
    - https://github.com/noahshinn/reflexion-draft
    - https://github.com/noahshinn/reflexion
"""
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.eval.reflexion import EM
from discussion_agents.cog.functional.react import _is_halted
from discussion_agents.cog.functional.reflexion import (
    _prompt_cot_agent,
    _prompt_react_agent,
    _truncate_scratchpad,
)
from discussion_agents.cog.modules.memory.reflexion import ReflexionMemory
from discussion_agents.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from discussion_agents.cog.prompts.react import REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES
from discussion_agents.cog.prompts.reflexion import (
    REFLEXION_COT_FEWSHOT_EXAMPLES,
    REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
    REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES,
    REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES_NO_CONTEXT,
    REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES,
)
from discussion_agents.utils.parse import parse_action, remove_newline


class ReflexionCoTAgent(BaseAgent):
    """Reflexion with Chain-of-Thought actor.

    Attributes:
        self_reflect_llm (BaseChatModel): The language model used for self-reflection.
        action_llm (BaseChatModel): The language model used for generating thoughts/actions.
        memory (Optional[ReflexionMemory]): An optional memory module to store the agent's internal state.
        reflector (Optional[ReflexionReflector]): An optional reflector module for guided self-reflection.
        max_reflections (int): An int specifying the max number of reflections to use in a subsequent run. Defaults to 3.
        max_trials (int): Max number of answering attempts before stopping generation. Must be greater than 1 for reflection to occur. Defaults to 1.
        patience (int): The number of incorrect retries before stopping. Must be >= 1 and <= max_trials. Defaults to max_trials.

    Methods:
        generate(context, question, key, strategy): Generates a response based on the given context, question, and strategy.
        reflect(context, question, strategy): Reflects on the previous response and modifies the strategy accordingly.
        retrieve(): Retrieves the current memory state of the agent.
        reset(): Resets the agent's state for a new problem-solving session.
    """

    def __init__(
        self,
        self_reflect_llm: BaseChatModel,
        action_llm: BaseChatModel,
        memory: Optional[ReflexionMemory] = None,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 1,
        patience: Optional[int] = None,
    ) -> None:
        """Initialization with default or provided values."""
        super().__init__()

        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm

        if not memory:
            memory = ReflexionMemory()
        self.memory = memory

        self.max_reflections = max_reflections
        if not reflector:
            reflector = ReflexionCoTReflector(
                llm=self_reflect_llm, max_reflections=max_reflections
            )
        self.reflector = reflector

        self.max_trials = max_trials
        if not patience:
            patience = max_trials
        self.patience = patience
        assert self.patience >= 1 and self.patience <= max_trials

        self._trial_n = 0
        self._finished = False
        self._answer = ""

    def generate(
        self,
        question: str,
        key: str,
        context: Optional[str] = None,
        strategy: Optional[str] = None,
        reset: bool = True,
    ) -> List[Tuple[bool, str, str]]:
        """Generates a response based on the provided context, question, and key.

        The `generate` method internally calls reflect (if possible), resets the memory,
        and generates a thought, action, and the observation (Finish).

        Args:
            question (str): The question to answer.
            key (str): The key to evaluate the correctness of the answer.
            context (Optional[str]): The context or background information. Defaults to None.
            strategy (Optional[str]): The strategy to use for reflection. Defaults to None.
            reset (bool): Resets the agent's memory. Defaults to True.

        Returns:
            result (List[bool, str, str]): A list of tuples containing (is_correct, answer, output)
                 from the ReflexionCoTAgent.
        """
        # Reset.
        if reset:
            self.reset()

        patience_cnt = 0
        result = []
        while not EM(self._answer, key) and self._trial_n < self.max_trials:
            # Reflect if possible.
            reflections_str = ""
            if self._trial_n > 0 and not EM(self._answer, key) and strategy:
                reflections_str = self.reflect(strategy, question, context)

            out = ""

            # Think.
            self.memory.add_memories("\nThought:")
            thought = _prompt_cot_agent(
                llm=self.action_llm,
                examples=REFLEXION_COT_FEWSHOT_EXAMPLES
                if context
                else REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
                reflections=reflections_str,
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
                reflections=reflections_str,
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

            self._trial_n += 1
            is_correct = EM(self._answer, key)
            result.append((is_correct, self._answer, out))

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == self.patience:
                break

        return result

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
        self.reflector.clear()
        self._finished = False
        self._trial_n = 0
        self._answer = ""


class ReflexionReActAgent(BaseAgent):
    """Reflexion with ReAct actor.

    Attributes:
        self_reflect_llm (BaseChatModel): The language model used for self-reflection.
        action_llm (BaseChatModel): The language model used for generating thoughts/actions.
        memory (Optional[ReflexionMemory]): An optional memory module to store the agent's internal state.
        reflector (Optional[ReflexionReflector]): An optional reflector module for guided self-reflection.
        max_reflections: (int): An int specifying the max number of reflections to use in a subsequent run. Defaults to 3.
        max_steps (int): Max number of steps for ReAct actor to take. Defaults to 6.
        max_tokens (int): Max tokens before the agent's memory is truncated. Defaults to 3896.
        max_trials (int): Max number of answering attempts before stopping generation. Must be greater than 1 for reflection to occur. Defaults to 1.
        patience (int): The number of incorrect retries before stopping. Must be >= 1 and <= max_trials. Defaults to max_trials.
        docstore (DocstoreExplorer): The Wikipedia docstore explorer.
        enc (Encoding): tiktoken Encoding for tracking token count of prompts.

    Methods:
        generate(context, question, key, strategy): Generates a response based on the given context, question, and strategy.
        reflect(context, question, strategy): Reflects on the previous response and modifies the strategy accordingly.
        retrieve(): Retrieves the current memory state of the agent.
        reset(): Resets the agent's state for a new problem-solving session.
    """

    def __init__(
        self,
        self_reflect_llm: BaseChatModel,
        action_llm: BaseChatModel,
        memory: Optional[ReflexionMemory] = None,
        reflector: Optional[ReflexionReActReflector] = None,
        max_reflections: int = 3,
        max_steps: int = 6,
        max_tokens: int = 3896,
        max_trials: int = 1,
        patience: Optional[int] = None,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__()
        self.self_reflect_llm = self_reflect_llm
        self.action_llm = action_llm

        if not memory:
            self.memory = ReflexionMemory()
        else:
            self.memory = memory

        self.max_reflections = max_reflections
        if not reflector:
            self.reflector = ReflexionReActReflector(
                llm=self_reflect_llm, max_reflections=max_reflections
            )
        else:
            self.reflector = reflector

        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_trials = max_trials

        if not patience:
            self.patience = max_trials
        else:
            self.patience = patience
        assert self.patience >= 0 and self.patience <= max_trials

        self.docstore = docstore
        self.enc = enc

        # Private variables.
        self._trial_n = 1
        self._step_n = 1
        self._finished = False
        self._answer = ""

    def generate(
        self,
        question: str,
        key: str,
        strategy: Optional[str] = None,
        reset: bool = True,
    ) -> List[Tuple[bool, str, str]]:
        """Processes a given question through ReAct and reflects using Reflexion strategies when possible.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            key (str): The answer to the question.
            strategy (Optional[str]): The reflection strategy. Can be of 3 types. Defaults to None.
                - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
                - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
                - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
            reset (bool): Whether to reset the internal state before processing. Defaults to True.

        Returns:
            result (List[Tuple[bool, str, str]]): List of outputs in the format (is_correct, answer, output)
                the ReflexionReActAgent.
        """
        # Reset.
        if reset:
            self.reset()

        patience_cnt = 0
        result = []
        while not EM(self._answer, key) and self._trial_n < self.max_trials + 1:
            # Reflect if possible.
            if (
                _is_halted(
                    finished=self._finished,
                    step_n=self._step_n,
                    max_steps=self.max_steps,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    max_tokens=self.max_tokens,
                    enc=self.enc,
                )
                and not EM(self._answer, key)
                and strategy
            ):
                self.reflect(strategy, question)

            out = ""
            self._step_n = 1
            self._finished = False
            self._answer = ""
            while not _is_halted(
                finished=self._finished,
                step_n=self._step_n,
                max_steps=self.max_steps,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                max_tokens=self.max_tokens,
                enc=self.enc,
            ):
                # Think.
                self.memory.add_memories("\nThought:")
                thought = _prompt_react_agent(
                    llm=self.action_llm,
                    examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
                    reflections=self.reflector.reflections_str,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    max_steps=self.max_steps,
                ).split("Action")[0]
                self.memory.add_memories(" " + thought)
                out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]

                # Act.
                self.memory.add_memories("\nAction:")
                action = _prompt_react_agent(
                    llm=self.action_llm,
                    examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
                    reflections=self.reflector.reflections_str,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    max_steps=self.max_steps,
                ).split("Observation")[0]
                self.memory.add_memories(" " + action)
                action_type, query = parse_action(action)
                out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]

                # Observe.
                self.memory.add_memories(f"\nObservation {self._step_n}: ")
                if action_type.lower() == "finish":
                    self._answer = query
                    self._finished = True
                    self.memory.add_memories(query)
                elif action_type.lower() == "search":
                    try:
                        self.memory.add_memories(
                            remove_newline(self.docstore.search(query))
                        )
                    except Exception:
                        self.memory.add_memories(
                            "Could not find that page, please try again."
                        )

                elif action_type.lower() == "lookup":
                    try:
                        self.memory.add_memories(
                            remove_newline(self.docstore.lookup(query))
                        )
                    except ValueError:
                        self.memory.add_memories(
                            "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
                        )
                else:
                    self.memory.add_memories(
                        "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
                    )

                self._step_n += 1
                out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]

            is_correct = EM(self._answer, key)
            result.append((is_correct, self._answer, out))

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == self.patience:
                break

            self._trial_n += 1

        return result

    def reflect(self, strategy: str, question: str) -> str:
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

        Returns:
            str: Generated reflections based on the strategy.
        """
        _, reflections_str = self.reflector.reflect(
            strategy=strategy,
            examples=REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES,
            question=question,
            scratchpad=_truncate_scratchpad(
                scratchpad=self.memory.load_memories()["scratchpad"], tokenizer=self.enc
            ),
        )

        return reflections_str

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()

    def reset(self) -> None:
        """Resets the internal state of the ReflexionReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self._trial_n = 1
        self._step_n = 1
        self._finished = False
        self._answer = ""
        self.memory.clear()
        self.reflector.clear()