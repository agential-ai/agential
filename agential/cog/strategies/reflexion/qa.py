"""Reflexion Agent strategies for QA."""

import re

from typing import Any, Dict, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.eval.reflexion import EM
from agential.cog.functional.reflexion import _prompt_cot_agent
from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
)
from agential.cog.strategies.reflexion.base import ReflexionCoTBaseStrategy
from agential.utils.parse import remove_newline


# def parse_qa_action(string: str) -> Tuple[str, str]:
#     """Parses an action string into an action type and its argument.

#     This method is used in ReAct and Reflexion.

#     Args:
#         string (str): The action string to be parsed.

#     Returns:
#         Tuple[str, str]: A tuple containing the action type and argument.
#     """
#     pattern = r"^(\w+)\[(.+)\]$"
#     match = re.match(pattern, string)

#     if match:
#         action_type = match.group(1)
#         argument = match.group(2)
#     else:
#         action_type = ""
#         argument = ""
#     return action_type, argument


class ReflexionCoTQAStrategy(ReflexionCoTBaseStrategy):
    """A strategy class for QA benchmarks using the ReflexionCoT agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionCoTReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 1.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 1,
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.llm = llm
        self.max_reflections = max_reflections
        self.max_trials = max_trials

        if not reflector:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        self.reflector = reflector

        self._scratchpad = ""
        self._finished = False
        self._answer = ""

    def generate(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generates an answer based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the answer.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            str: The generated answer.
        """
        self._scratchpad += "\nA:"
        answer = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        answer = remove_newline(answer)
        self._scratchpad += " " + answer

        return answer

    # def generate_action(
    #     self,
    #     question: str,
    #     examples: str,
    #     reflections: str,
    #     prompt: str,
    #     additional_keys: Dict[str, str],
    #     **kwargs: Dict[str, Any],
    # ) -> str:
    #     """Generates an action based on the question, examples, and prompt.

    #     Args:
    #         question (str): The question to be answered.
    #         examples (str): Examples to guide the generation process.
    #         reflections (str): Reflections to consider during generation.
    #         prompt (str): The prompt used for generating the action.
    #         additional_keys (Dict[str, str]): Additional keys for the generation process.
    #         **kwargs (Dict[str, Any]): Additional arguments.

    #     Returns:
    #         str: The generated query.
    #     """
    #     self._scratchpad += "\nAction:"
    #     action = _prompt_cot_agent(
    #         llm=self.llm,
    #         examples=examples,
    #         reflections=reflections,
    #         question=question,
    #         scratchpad=self._scratchpad,
    #         prompt=prompt,
    #         additional_keys=additional_keys,
    #     )
    #     action = remove_newline(action).strip()
    #     query = action.split("So the answer is:")[-1]

    #     self._scratchpad += " " + action

    #     return query

    def generate_observation(
        self, query: str, key: str
    ) -> bool:
        """Generates an observation based on the action type and query.

        Args:
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[bool, str]: The generated observation.
        """
        self._scratchpad += f"\nObservation: "
        self._finished = True
        self._answer = query

        self._scratchpad += "Answer is CORRECT" if EM(self._answer, key) else "Answer is INCORRECT"

        return EM(self._answer, key)

    def create_output_dict(
        self, thought: str, query: str, obs: str, key: str
    ) -> Dict[str, str]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            query (str): The query for the action.
            obs (str): The generated observation.
            key (str): The key for the observation.

        Returns:
            Dict[str, str]: A dictionary containing the thought, action type, query, and observation.
        """
        return {
            "thought": thought,
            "query": query,
            "obs": obs,
            "answer": self._answer,
            "is_correct": EM(self._answer, key),
        }

    def halting_condition(
        self,
        idx: int,
        key: str,
        **kwargs: Dict[str, Any],
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            key (str): The key for the observation.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        max_trials = kwargs.get("max_trials", self.max_trials)
        return not EM(self._answer, key) and idx < max_trials

    def reset(self, **kwargs: Dict[str, Any]) -> None:
        """Resets the internal state of the strategy.

        Resets the scratchpad and the finished flag.
        Resets only the scratchpad if specified with 'only_scratchpad'.

        Args:
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            None
        """
        only_scratchpad = kwargs.get("only_scratchpad", False)
        if only_scratchpad:
            self._scratchpad = ""
        else:
            self.reflector.clear()
            self._scratchpad = ""
            self._finished = False
            self._answer = ""

    def reflect(
        self,
        reflection_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> str:
        """Reflects on a given question, context, examples, prompt, and additional keys using the specified reflection strategy.

        Args:
            reflection_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            str: The reflection string.
        """
        _, reflections_str = self.reflector.reflect(
            reflection_strategy=reflection_strategy,
            examples=examples,
            question=question,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        return reflections_str

    def should_reflect(
        self,
        idx: int,
        reflection_strategy: str,
        key: str,
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            idx (int): The current step.
            reflection_strategy (str): The strategy to use for reflection.
            key (str): The key for the observation.

        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        return idx > 0 and not EM(self._answer, key) and reflection_strategy


class ReflexionCoTHotQAStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTTriviaQAStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTAmbigNQStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTFEVERStrategy(ReflexionCoTQAStrategy):
    """A strategy class for the FEVER benchmark using the ReflexionCoT agent."""

    pass
