"""Reflexion Agent strategies for QA."""

import re

from typing import Any, Dict, Optional, Tuple, List

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.eval.reflexion import EM
from agential.cog.functional.reflexion import _prompt_cot_agent
from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
)
from agential.cog.strategies.reflexion.base import ReflexionCoTBaseStrategy, ReflexionReActBaseStrategy
from agential.utils.parse import remove_newline
import tiktoken

from langchain_community.docstore.wikipedia import Wikipedia

from tiktoken import Encoding

from agential.utils.docstore import DocstoreExplorer
from agential.cog.functional.reflexion import (
    _is_halted,
    _truncate_scratchpad,
    _prompt_react_agent,
)

def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct and Reflexion.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


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
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            str: The generated thought.
        """
        self._scratchpad += "\nThought:"
        thought = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(thought).split("Action")[0]
        self._scratchpad += " " + thought

        return thought

    def generate_action(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        self._scratchpad += "\nAction:"
        action = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = remove_newline(action).strip()
        self._scratchpad += " " + action
        action_type, query = parse_qa_action(action)

        return action_type, query

    def generate_observation(
        self, action_type: str, query: str, key: str
    ) -> Tuple[bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[bool, str]: The generated observation.
        """
        self._scratchpad += f"\nObservation: "
        if action_type.lower() == "finish":
            self._finished = True
            self._answer = query
            if EM(self._answer, key):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        else:
            obs = "Invalid action type, please try again."
        self._scratchpad += obs

        return EM(self._answer, key), obs

    def create_output_dict(
        self, thought: str, action_type: str, obs: str, is_correct: bool
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            obs (str): The generated observation.
            is_correct (bool): Whether the answer is correct.

        Returns:
            Dict[str, str]: A dictionary containing the thought, action type, observation, answer, and is_correct.
        """
        return {
            "thought": thought,
            "action_type": action_type,
            "obs": obs,
            "answer": self._answer,
            "is_correct": is_correct,
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
        return EM(self._answer, key) or idx >= max_trials

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
            self.reflector.reset()
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
            question=question,
            examples=examples,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        return reflections_str

    def reflect_condition(
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


class ReflexionReActQAStrategy(ReflexionReActBaseStrategy):
    """A strategy class for QA benchmarks using the ReflexionReAct agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionCoTReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 1.
        max_steps (int): The maximum number of steps allowed. Defaults to 6.
        max_tokens (int): The maximum number of tokens allowed. Defaults to 3896.
        docstore (DocstoreExplorer): The document store explorer for retrieving relevant documents. Defaults to Wikipedia.
        enc (Encoding): The encoding for tokenization. Defaults to gpt-3.5-turbo.
    """
    def __init__(
        self, 
        llm: BaseChatModel,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 1,
        max_steps: int = 6,
        max_tokens: int = 3896,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.max_reflections = max_reflections
        self.max_trials = max_trials

        if not reflector:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        self.reflector = reflector

        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        self._finished = False
        self._answer = ""
        self._scratchpad = ""

    def generate(
        self, 
        question: str, 
        examples: str, 
        reflections: str, 
        prompt: str, 
        additional_keys: Dict[str, str], 
        **kwargs: Dict[str, Any]
    ) -> str:
        max_steps = kwargs.get("max_steps", self.max_steps)  # type: ignore

        self._scratchpad += "\nThought:"
        thought = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            reflections=reflections,
            scratchpad=self._scratchpad,
            max_steps=max_steps,  # type: ignore
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(thought).split("Action")[0]
        self._scratchpad += " " + thought

        return thought
    
    def generate_action(
        self, 
        question: str, 
        examples: str, 
        reflections: str,
        prompt: str, 
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> Tuple[str, str]:
        max_steps = kwargs.get("max_steps", self.max_steps)
        self._scratchpad += "\nAction:"
        action = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            reflections=reflections,
            scratchpad=self._scratchpad,
            max_steps=max_steps,  # type: ignore
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = remove_newline(action).split("Observation")[0]
        self._scratchpad += " " + action
        action_type, query = parse_qa_action(action)

        return action_type, query


    def generate_observation(
        self, 
        step_idx: int, 
        action_type: str, 
        query: str,
        key: str,
    ) -> Tuple[bool, str]:
        self._scratchpad += f"\nObservation {step_idx}: "
        if action_type.lower() == "finish":
            self._answer = query
            self._finished = True
            if EM(self._answer, key):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        elif action_type.lower() == "search":
            try:
                obs = remove_newline(self.docstore.search(query))
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                obs = remove_newline(self.docstore.lookup(query))
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        self._scratchpad += obs

        return EM(self._answer, key), obs

    def create_output_dict(
        self, 
        react_out: List[Dict[str, Any]],
        reflections: str,
    ) -> Dict[str, str]:
        return {
            "react_output": react_out,
            "reflections": reflections,
        }

    def react_create_output_dict(
        self, 
        thought: str, 
        action_type: str, 
        query: str, 
        obs: str, 
        is_correct: bool
    ) -> Dict[str, Any]:
        return {
            "thought": thought,
            "action_type": action_type,
            "query": query,
            "observation": obs,
            "is_correct": is_correct, 
        }
    
    def halting_condition(self, idx: int, key: str, **kwargs: Dict[str, Any]) -> bool:
        max_trials = kwargs.get("max_trials", self.max_trials)
        return not EM(self._answer, key) and idx < max_trials + 1
    
    def react_halting_condition(
        self,
        step_idx: int,
        question: str, 
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> bool:
        max_steps = kwargs.get("max_steps", self.max_steps)

        return _is_halted(
            finished=self._finished,
            step_idx=step_idx,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            reflections=reflections,
            max_steps=max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self, **kwargs: Dict[str, Any]) -> None:
        no_reflector = kwargs.get('no_reflector', False)
        if not no_reflector:
            self.reflector.reset()
        self._scratchpad = ""
        self._finished = False
        self._answer = ""    

    def reflect(
        self, 
        reflection_strategy: str, 
        question: str, 
        examples: str, 
        prompt: str, 
        additional_keys: Dict[str, str]
    ) -> str:
        _, reflections_str = self.reflector.reflect(
            strategy=reflection_strategy,
            question=question,
            examples=examples,
            scratchpad=_truncate_scratchpad(
                scratchpad=self._scratchpad, 
                tokenizer=self.enc
            ),
            prompt=prompt,
            additional_keys=additional_keys,
        )

        return reflections_str

    def reflect_condition(
        self, 
        step_idx: int, 
        reflection_strategy: str, 
        question: str, 
        examples: str, 
        key: str, 
        prompt: str, 
        additional_keys: Dict[str, str], 
        **kwargs: Dict[str, str]
    ) -> bool:
        max_steps = kwargs.get("max_steps", self.max_steps)

        halted =  _is_halted(
            finished=self._finished,
            step_idx=step_idx,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            reflections=self.reflector.reflections_str,
            max_steps=max_steps,  # type: ignore
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        return halted and not EM(self._answer, key) and reflection_strategy
    



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


class ReflexionReActHotQAStrategy(ReflexionReActQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActTriviaQAStrategy(ReflexionReActQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActAmbigNQStrategy(ReflexionReActQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActFEVERStrategy(ReflexionReActQAStrategy):
    """A strategy class for the FEVER benchmark using the ReflexionReAct agent."""

    pass