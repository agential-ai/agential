"""Reflexion general strategy."""

from typing import Any, Dict, List, Optional, Tuple

from agential.cog.reflexion.functional import _prompt_cot_agent
from agential.cog.reflexion.reflect import ReflexionCoTReflector
from agential.cog.reflexion.strategies.base import ReflexionCoTBaseStrategy
from agential.llm.llm import BaseLLM
from agential.utils.metrics import PromptMetrics, get_token_cost_time
from agential.utils.parse import remove_newline


class ReflexionCoTGeneralStrategy(ReflexionCoTBaseStrategy):
    """A general strategy class for the ReflexionCoT agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionCoTReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 3.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 3,
        testing: bool = False
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        super().__init__(llm=llm, reflector=reflector, max_reflections=max_reflections, max_trials=max_trials, testing=testing)

    def generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, PromptMetrics]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, PromptMetrics]: The updated scratchpad, the generated thought, and the metrics for the thought.
        """
        scratchpad += f"\nThought {idx}: "
        out = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = out.choices[0].message.content
        thought = remove_newline(thought).split("Action")[0].strip()
        scratchpad += thought

        return scratchpad, thought, get_token_cost_time(out)