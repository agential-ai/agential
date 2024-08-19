"""Reflexion general strategy."""

from typing import Any, Dict, List, Optional, Tuple

from agential.cog.react.output import ReActOutput
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

    def generate(
        self, 
        question: str,
        key: str,
        examples: str,
        prompt: str,
        reflect_examples: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        patience: int,
        reset: bool,
    ) -> ReActOutput:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            key (str): The key for the output.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt to guide the generation process.
            reflect_examples (str): Examples to guide the reflection process.
            reflect_prompt (str): The prompt to guide the reflection process.
            reflect_strategy (str): The strategy to use for reflection.
            additional_keys (Dict[str, str]): Additional keys to include in the output.
            reflect_additional_keys (Dict[str, str]): Additional keys to include in the reflection output.
            patience (int): The patience level for the agent.
            reset (bool): Whether to reset the agent.

        Returns:
            ReActOutput: The output of the agent.
        """
        if reset:
            self.reset()

        scratchpad = ""
        answer = ""
        finished = False
        idx, patience_cnt = 0, 0
        out = []
        while not self.halting_condition(idx=idx, key=key):
            # Reflect if possible.
            reflections: List[str] = []
            reflections_str = ""
            if self.reflect_condition(
                idx=idx,
                reflect_strategy=reflect_strategy,
                key=key,
            ):
                reflections, reflections_str = self.reflect(
                    reflect_strategy=reflect_strategy,
                    question=question,
                    examples=reflect_examples,
                    prompt=reflect_prompt,
                    additional_keys=reflect_additional_keys,
                )

            scratchpad = ""

            # Think.
            scratchpad, thought, thought_metrics = self.generate_thought(
                idx=idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections_str,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_metrics = self.generate_action(
                idx=idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections_str,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, is_correct, obs, finished = self.generate_observation(
                idx=idx,
                scratchpad=scratchpad,
                action_type=action_type,
                query=query,
                key=key,
            )

            out.append(
                ReflexionCoTStepOutput(
                    **self.strategy.create_output_dict(
                        thought=thought,
                        action_type=action_type,
                        obs=obs,
                        is_correct=is_correct,
                        reflections=reflections,
                    )
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        return out

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
            idx (int): The index of the thought.
            scratchpad (str): The scratchpad containing previous thoughts.
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

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptMetrics]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            idx (int): The current index of the action.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str, str, PromptMetrics]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
        """
        raise NotImplementedError
    
    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, str, bool]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, bool, str, bool]: The updated scratchpad, the answer, a boolean indicating if the observation is correct, the observation itself, and a boolean indicating if the observation is finished.
        """
        raise NotImplementedError
    
    def halting_condition(
        self,
        idx: int,
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            key (str): The key for the observation.
            answer (str): The answer generated.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError
    
    def reflect_condition(
        self,
        idx: int,
        reflect_strategy: Optional[str],
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            idx (int): The current step.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            key (str): The key for the observation.
            answer (str): The answer generated.
            
        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        raise NotImplementedError

    def reflect(
        self,
        scratchpad: str,
        reflect_strategy: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[List[str], str, PromptMetrics]:
        """Reflects on a given question, context, examples, prompt, and additional keys using the specified reflection strategy.

        Args:
            scratchpad (str): The scratchpad containing previous reflections.
            reflect_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str, PromptMetrics]: The reflections, the reflection string, and the metrics.
        """
        reflections, reflections_str, reflections_out = self.reflector.reflect(
            reflect_strategy=reflect_strategy,
            question=question,
            examples=examples,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        reflection_metrics = (
            get_token_cost_time(reflections_out) if reflections_out else None
        )
        return reflections, reflections_str, reflection_metrics

    def reset(self) -> None:
        """Resets the internal state of the strategy."""
        self.reflector.reset()
