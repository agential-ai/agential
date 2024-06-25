"""Reflexion Agent strategies for Math."""

import re

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.eval.reflexion import EM
from agential.cog.functional.reflexion import (
    _is_halted,
    _prompt_cot_agent,
    _prompt_react_agent,
    _truncate_scratchpad,
)
from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)
from agential.utils.general import safe_execute
from agential.utils.parse import remove_newline


def parse_math_action(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    match = re.search(r"\b(Finish)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


class ReflexionCoTMathStrategy(ReflexionCoTBaseStrategy):
    """A strategy class for Math benchmarks using the ReflexionCoT agent.

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
        **kwargs: Any,
    ) -> str:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

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
        thought = remove_newline(thought).split("Action")[0].strip().split("\n")[0]
        self._scratchpad += " " + thought

        return thought

    def generate_action(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str]:
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
        action = action.split("Observation")[0].strip()

        action_type, query = parse_math_action(action)
        self._scratchpad += f" {action_type}[\n```python\n{query}\n```\n]"

        return action_type, query

    def generate_observation(
        self, action_type: str, query: str, key: str
    ) -> Tuple[bool | str]:
        self._scratchpad += f"\nObservation: "
        if action_type.lower() == "finish":
            self._finished = True
            self._answer = query
            answer, _ = safe_execute(self._answer)
            if EM(answer[0], key):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        else:
            obs = "Invalid action type, please try again."
        self._scratchpad += obs

        return EM(answer[0], key), obs

    def create_output_dict(
        self,
        thought: str,
        action_type: str,
        obs: str,
        is_correct: bool,
        reflections: List[str],
    ) -> Dict[str, Any]:
        return {
            "thought": thought,
            "action_type": action_type,
            "observation": obs,
            "answer": self._answer,
            "is_correct": is_correct,
            "reflections": reflections,
        }

    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        max_trials = kwargs.get("max_trials", self.max_trials)
        return EM(self._answer, key) or idx >= max_trials
    
    def reset(self, *args: Any, **kwargs: Any) -> None:
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
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str] | str]:
        reflections, reflections_str = self.reflector.reflect(
            reflect_strategy=reflect_strategy,
            question=question,
            examples=examples,
            scratchpad=self._scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        return reflections, reflections_str

    def reflect_condition(
        self, idx: int, reflect_strategy: str | None, key: str
    ) -> bool:
        return idx > 0 and not EM(self._answer, key) and reflect_strategy is not None


class ReflexionReActMathStrategy(ReflexionReActBaseStrategy):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return super().generate(*args, **kwargs)

    def generate_action(
        self,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> Tuple[str]:
        return super().generate_action(
            question, examples, reflections, prompt, additional_keys, **kwargs
        )

    def generate_observation(
        self, step_idx: int, action_type: str, query: str, key: str
    ) -> Tuple[bool | str]:
        return super().generate_observation(step_idx, action_type, query, key)

    def create_output_dict(
        self, react_out: List[Dict[str, Any]], reflections: List[str]
    ) -> Dict[str, Any]:
        return super().create_output_dict(react_out, reflections)

    def react_create_output_dict(
        self, thought: str, action_type: str, query: str, obs: str, is_correct: bool
    ) -> Dict[str, str]:
        return super().react_create_output_dict(
            thought, action_type, query, obs, is_correct
        )

    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        return super().halting_condition(idx, key, **kwargs)

    def react_halting_condition(
        self,
        step_idx: int,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> bool:
        return super().react_halting_condition(
            step_idx, question, examples, reflections, prompt, additional_keys, **kwargs
        )

    def reset(self, *args: Any, **kwargs: Any) -> None:
        return super().reset(*args, **kwargs)

    def reflect(
        self,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str] | str]:
        return super().reflect(
            reflect_strategy, question, examples, prompt, additional_keys
        )

    def reflect_condition(
        self,
        step_idx: int,
        reflect_strategy: str | None,
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, str],
    ) -> bool:
        return super().reflect_condition(
            step_idx,
            reflect_strategy,
            question,
            examples,
            key,
            prompt,
            additional_keys,
            **kwargs,
        )


class ReflexionCoTGSM8KStrategy(ReflexionCoTMathStrategy):
    """A strategy class for the GSM8K benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTSVAMPStrategy(ReflexionCoTMathStrategy):
    """A strategy class for the SVAMP benchmark using the ReflexionCoT agent."""

    pass


class ReflexionCoTTabMWPStrategy(ReflexionCoTMathStrategy):
    """A strategy class for the TabMWP benchmark using the ReflexionCoT agent."""

    pass


class ReflexionReActGSM8KStrategy(ReflexionReActMathStrategy):
    """A strategy class for the GSM8K benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActSVAMPStrategy(ReflexionReActMathStrategy):
    """A strategy class for the SVAMP benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActTabMWPStrategy(ReflexionReActMathStrategy):
    """A strategy class for the TabMWP benchmark using the ReflexionReAct agent."""

    pass
