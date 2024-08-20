"""Reflexion general strategy."""

import time

from typing import Any, Dict, List, Optional, Tuple

from tiktoken import Encoding
import tiktoken

from agential.cog.reflexion.functional import _is_halted, _prompt_cot_agent, _prompt_react_agent, _truncate_scratchpad, accumulate_metrics_cot
from agential.cog.reflexion.output import ReflexionCoTOutput, ReflexionCoTStepOutput, ReflexionReActOutput, ReflexionReActReActStepOutput, ReflexionReActStepOutput
from agential.cog.reflexion.reflect import ReflexionCoTReflector, ReflexionReActReflector
from agential.cog.reflexion.strategies.base import ReflexionCoTBaseStrategy, ReflexionReActBaseStrategy
from agential.eval.em import EM
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
        testing: bool = False,
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        super().__init__(
            llm=llm,
            reflector=reflector,
            max_reflections=max_reflections,
            max_trials=max_trials,
            testing=testing,
        )

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        patience: int,
        reset: bool,
    ) -> ReflexionCoTOutput:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            key (str): The key for the output.
            examples (str): Examples to guide the generation process.
            reflect_examples (str): Examples to guide the reflection process.
            prompt (str): The prompt to guide the generation process.
            reflect_prompt (str): The prompt to guide the reflection process.
            reflect_strategy (str): The strategy to use for reflection.
            additional_keys (Dict[str, str]): Additional keys to include in the output.
            reflect_additional_keys (Dict[str, str]): Additional keys to include in the reflection output.
            patience (int): The patience level for the agent.
            reset (bool): Whether to reset the agent.

        Returns:
            ReflexionCoTOutput: The output of the agent.
        """
        start = time.time()

        if reset:
            self.reset()

        scratchpad = ""
        answer = ""
        idx, patience_cnt = 0, 0
        steps = []
        while not self.halting_condition(idx=idx, key=key, answer=answer):
            # Reflect if possible.
            reflections: List[str] = []
            reflections_str = ""
            reflection_metrics: Optional[PromptMetrics] = None
            if self.reflect_condition(
                idx=idx,
                reflect_strategy=reflect_strategy,
                key=key,
                answer=answer,
            ):
                reflections, reflections_str, reflection_metrics = self.reflect(
                    scratchpad=scratchpad,
                    reflect_strategy=reflect_strategy,
                    question=question,
                    examples=reflect_examples,
                    prompt=reflect_prompt,
                    additional_keys=reflect_additional_keys,
                )

            scratchpad = ""

            # Think.
            scratchpad, thought, thought_metrics = self.generate_thought(
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections_str,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_metrics = self.generate_action(
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections_str,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, is_correct, obs = self.generate_observation(
                scratchpad=scratchpad,
                action_type=action_type,
                query=query,
                key=key,
            )

            steps.append(
                ReflexionCoTStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    is_correct=is_correct,
                    reflections=reflections,
                    thought_metrics=thought_metrics,
                    action_metrics=action_metrics,
                    reflection_metrics=reflection_metrics,
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        total_time = time.time() - start
        total_metrics = accumulate_metrics_cot(steps)
        out = ReflexionCoTOutput(
            answer=answer,
            total_prompt_tokens=total_metrics["total_prompt_tokens"],
            total_completion_tokens=total_metrics["total_completion_tokens"],
            total_tokens=total_metrics["total_tokens"],
            total_prompt_cost=total_metrics["total_prompt_cost"],
            total_completion_cost=total_metrics["total_completion_cost"],
            total_cost=total_metrics["total_cost"],
            total_prompt_time=total_metrics["total_prompt_time"],
            total_time=total_time if not self.testing else 0.5,
            additional_info=steps,
        )

        return out

    def generate_thought(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, PromptMetrics]:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            scratchpad (str): The scratchpad containing previous thoughts.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, PromptMetrics]: The updated scratchpad, the generated thought, and the metrics for the thought.
        """
        scratchpad += f"\nThought: "
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
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, PromptMetrics]:
        """Generates an action based on the question, examples, and prompt.

        Args:
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
        self, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, bool, str, bool]: The updated scratchpad, the answer, a boolean indicating if the observation is correct, and the observation itself.
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


class ReflexionReActGeneralStrategy(ReflexionReActBaseStrategy):
    """A general strategy class for the ReflexionReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionReActReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 3.
        max_steps (int): The maximum number of steps allowed. Defaults to 6.
        max_tokens (int): The maximum number of tokens allowed. Defaults to 5000.
        enc (Encoding): The encoding for tokenization. Defaults to gpt-3.5-turbo.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: Optional[ReflexionReActReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 3,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionReActReflector(
                llm=llm, max_reflections=max_reflections
            )
        super().__init__(
            llm=llm, 
            reflector=reflector, 
            max_reflections=max_reflections, 
            max_trials=max_trials, 
            max_steps=max_steps, 
            max_tokens=max_tokens, 
            enc=enc, 
            testing=testing,
        )

        self._finished = False
        self._answer = ""
        self._scratchpad = ""

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        reflect_examples: str,
        prompt: str,
        reflect_prompt: str,
        reflect_strategy: str,
        additional_keys: Dict[str, str],
        reflect_additional_keys: Dict[str, str],
        patience: int,
        reset: bool,        
    ) -> ReflexionReActOutput:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            key (str): The key for the output.
            examples (str): Examples to guide the generation process.
            reflect_examples (str): Examples to guide the reflection process.
            prompt (str): The prompt to guide the generation process.
            reflect_prompt (str): The prompt to guide the reflection process.
            reflect_strategy (str): The strategy to use for reflection.
            additional_keys (Dict[str, str]): Additional keys to include in the output.
            reflect_additional_keys (Dict[str, str]): Additional keys to include in the reflection output.
            patience (int): The patience level for the agent.
            reset (bool): Whether to reset the agent.

        Returns:
            ReflexionReActOutput: The output of the agent.
        """
        # Reset.
        if reset:
            self.reset()

        scratchpad = ""
        answer = ""
        finished = False
        idx, step_idx, patience_cnt = 1, 1, 0
        steps = []
        while not self.halting_condition(idx=idx, key=key, answer=answer):
            # Reflect if possible.
            reflections: List[str] = []
            reflections_str = ""
            if self.reflect_condition(
                answer=answer,
                finished=finished,
                idx=step_idx,
                scratchpad=scratchpad,
                reflect_strategy=reflect_strategy,
                question=question,
                examples=examples,
                key=key,
                prompt=prompt,
                additional_keys=additional_keys,
            ):
                reflections, reflections_str, reflection_metrics = self.reflect(
                    scratchpad=scratchpad,
                    reflect_strategy=reflect_strategy,
                    question=question,
                    examples=reflect_examples,
                    prompt=reflect_prompt,
                    additional_keys=reflect_additional_keys,
                )
                
            step_idx, is_correct, scratchpad, finished, answer, react_steps = self.generate_react(
                question=question,
                key=key,
                examples=examples,
                reflections=reflections_str,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            steps.append(
                ReflexionReActStepOutput(
                    steps=react_steps,
                    reflections=reflections,
                    reflection_metrics=reflection_metrics,
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        out = ReflexionReActOutput(
            answer=answer,
            total_prompt_tokens=,

            additional_info=steps
        )

        return out

    def generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
    ) -> Tuple[int, bool, str, bool, str, List[ReflexionReActReActStepOutput]]:
        """Generates a reaction based on the given question, key, examples, reflections, prompt, and additional keys.

        Args:
            question (str): The question to be answered.
            key (str): The key for the observation.
            examples (str): Examples to guide the reaction process.
            reflections (str): The reflections to guide the reaction process.
            prompt (str): The prompt or instruction to guide the reaction.
            additional_keys (Dict[str, str]): Additional keys for the reaction process.

        Returns:
            Tuple[int, bool, str, bool, str, List[ReflexionReActReActStepOutput]]: The reaction, whether the reaction is finished, the answer, whether the reaction is valid, the scratchpad, and the steps.
        """
       
        react_steps = []
        step_idx = 1
        scratchpad = ""
        finished = False
        answer = ""
        while not self.react_halting_condition(
            finished=finished,
            idx=step_idx,
            scratchpad=scratchpad,
            question=question,
            examples=examples,
            reflections=reflections,
            prompt=prompt,
            additional_keys=additional_keys,
        ):
            # Think.
            scratchpad, thought, thought_metrics = self.generate_thought(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_metrics = self.generate_action(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                reflections=reflections,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, finished, is_correct, obs, external_tool_info = self.generate_observation(
                idx=step_idx,
                scratchpad=scratchpad,
                action_type=action_type,
                query=query,
                key=key,
            )

            react_steps.append(
                ReflexionReActReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    obs=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    is_correct=is_correct,
                    thought_metrics=thought_metrics,
                    action_metrics=action_metrics,
                )
            )

            step_idx += 1

        return step_idx, is_correct, scratchpad, finished, answer, react_steps


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
        """Generates a thought based on the given question, examples, reflections, prompt, and additional keys.

        Args:
            idx (int): The current step.
            scratchpad (str): The scratchpad containing previous thoughts and reflections.
            question (str): The question to generate a thought for.
            examples (str): Examples to guide the thought generation process.
            reflections (str): Reflections to consider during the thought generation process.
            prompt (str): The prompt or instruction to guide the thought generation.
            additional_keys (Dict[str, str]): Additional keys for the thought generation process.

        Returns:
            Tuple[str, str, PromptMetrics]: The updated scratchpad, the generated thought, and the thought metrics.
        """

        scratchpad += f"\nThought {idx}: "
        out = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            reflections=reflections,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
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
        """Generate an action for the current step in the reasoning process.

        Args:
            idx (int): The current step index.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, PromptMetrics]: A tuple containing the updated trajectory, action type, query, and the metrics.
        """
        raise NotImplementedError
    
    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, bool, str, Dict[str, Any]]:
        """Generate an observation based on the given inputs.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed.
            query (str): The query or action to observe.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: A tuple containing:
                - The updated scratchpad.
                - The answer.
                - A boolean indicating if finished.
                - The generated observation.
                - A boolean indicating if the task is finished.
                - The observation.
                - A dictionary with additional information.
        """
        raise NotImplementedError
    
    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, bool, str, Dict[str, Any]]:
        """Generate an observation based on the given inputs.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed.
            query (str): The query or action to observe.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: A tuple containing:
                - The updated scratchpad.
                - The answer.
                - A boolean indicating if finished.
                - The generated observation.
                - A boolean indicating if the task is finished.
                - The observation.
                - A dictionary with additional information.
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
    
    def react_halting_condition(
        self,
        finished: bool,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the halting condition has been met in the ReflexionReAct agent.

        Args:
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The question to generate an action for.
            examples (str): Examples to guide the action generation process.
            reflections (str): Reflections to consider during the action generation process.
            prompt (str): The prompt or instruction to guide the action generation.
            additional_keys (Dict[str, str]): Additional keys for the action generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise. The halting condition is met when the answer is not correct and the current step index is less than the maximum number of steps plus one.
        """

        return _is_halted(
            finished=finished,
            step_idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            reflections=reflections,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reflect_condition(
        self,
        answer: str,
        finished: bool,
        idx: int,
        scratchpad: str,
        reflect_strategy: Optional[str],
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the reflection condition has been met in the ReflexionReAct agent.

        Args:
            answer (str): The answer generated.
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            key (str): The key for the observation.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            bool: True if the reflection condition is met, False otherwise. The reflection condition is met when the agent is halted, the answer is not correct, and the reflection strategy is provided.
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
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            reflect_strategy (str): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            Tuple[List[str], str, PromptMetrics]: The reflections, reflection string, and the metrics for the reflection process.
        """
        reflections, reflections_str, reflections_out = self.reflector.reflect(
            reflect_strategy=reflect_strategy,
            question=question,
            examples=examples,
            scratchpad=_truncate_scratchpad(
                scratchpad=scratchpad, tokenizer=self.enc
            ),
            prompt=prompt,
            additional_keys=additional_keys,
        )
        reflection_metrics = get_token_cost_time(reflections_out) if reflections_out else None

        return reflections, reflections_str, reflection_metrics

    def reset(self, **kwargs: Any) -> None:
        """Resets the internal state of the strategy.

        Resets the scratchpad and the finished flag.
        Resets only the scratchpad if specified with 'only_scratchpad'.

        Args:
            **kwargs (Any): Additional keyword arguments.
        """
        no_reflector = kwargs.get("no_reflector", False)
        if not no_reflector:
            self.reflector.reset()
        self._scratchpad = ""
        self._finished = False
        self._answer = ""
