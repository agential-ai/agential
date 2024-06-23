"""Reflexion Agent strategies for Math."""

from typing import Any, Dict, List, Tuple, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from agential.cog.strategies.reflexion.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
)
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
from agential.utils.parse import remove_newline


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
        thought = remove_newline(thought).split("Action")[0]
        self._scratchpad += " " + thought

        return thought
    
    def generate_action(self, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[str, str]:
        return super().generate_action(question, examples, reflections, prompt, additional_keys)
    
    def generate_observation(self, action_type: str, query: str, key: str) -> Tuple[bool | str]:
        return super().generate_observation(action_type, query, key)
    
    def create_output_dict(self, thought: str, action_type: str, obs: str, is_correct: bool, reflections: List[str]) -> Dict[str, Any]:
        return super().create_output_dict(thought, action_type, obs, is_correct, reflections)

    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        return super().halting_condition(idx, key, **kwargs)
    
    def reset(self, *args: Any, **kwargs: Any) -> None:
        return super().reset(*args, **kwargs)
    
    def reflect(self, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[List[str] | str]:
        return super().reflect(reflect_strategy, question, examples, prompt, additional_keys)
    
    def reflect_condition(self, idx: int, reflect_strategy: str | None, key: str) -> bool:
        return super().reflect_condition(idx, reflect_strategy, key)


class ReflexionReActMathStrategy(ReflexionReActBaseStrategy):
    def __init__(self, llm: BaseChatModel) -> None:
        super().__init__(llm)

    def generate(self, *args: Any, **kwargs: Any) -> str:
        return super().generate(*args, **kwargs)
    
    def generate_action(self, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str], **kwargs: Any) -> Tuple[str]:
        return super().generate_action(question, examples, reflections, prompt, additional_keys, **kwargs)
    
    def generate_observation(self, step_idx: int, action_type: str, query: str, key: str) -> Tuple[bool | str]:
        return super().generate_observation(step_idx, action_type, query, key)
    
    def create_output_dict(self, react_out: List[Dict[str, Any]], reflections: List[str]) -> Dict[str, Any]:
        return super().create_output_dict(react_out, reflections)
    
    def react_create_output_dict(self, thought: str, action_type: str, query: str, obs: str, is_correct: bool) -> Dict[str, str]:
        return super().react_create_output_dict(thought, action_type, query, obs, is_correct)
    
    def halting_condition(self, idx: int, key: str, **kwargs: Any) -> bool:
        return super().halting_condition(idx, key, **kwargs)
    
    def react_halting_condition(self, step_idx: int, question: str, examples: str, reflections: str, prompt: str, additional_keys: Dict[str, str], **kwargs: Any) -> bool:
        return super().react_halting_condition(step_idx, question, examples, reflections, prompt, additional_keys, **kwargs)
    
    def reset(self, *args: Any, **kwargs: Any) -> None:
        return super().reset(*args, **kwargs)
    
    def reflect(self, reflect_strategy: str, question: str, examples: str, prompt: str, additional_keys: Dict[str, str]) -> Tuple[List[str] | str]:
        return super().reflect(reflect_strategy, question, examples, prompt, additional_keys)
    
    def reflect_condition(self, step_idx: int, reflect_strategy: str | None, question: str, examples: str, key: str, prompt: str, additional_keys: Dict[str, str], **kwargs: Dict[str, str]) -> bool:
        return super().reflect_condition(step_idx, reflect_strategy, question, examples, key, prompt, additional_keys, **kwargs)
    

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
