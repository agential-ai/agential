"""CLIN general strategy."""

import time

from typing import Any, Dict, List, Optional, Tuple

from agential.agents.clin.functional import (
    _is_halted,
    _prompt_meta_summary,
    _prompt_react_agent,
    accumulate_metrics,
)
from agential.agents.clin.memory import CLINMemory
from agential.agents.clin.output import CLINOutput, CLINReActStepOutput, CLINStepOutput
from agential.agents.clin.strategies.base import CLINBaseStrategy
from agential.core.llm import BaseLLM, Response
from agential.utils.parse import remove_newline


class CLINGeneralStrategy(CLINBaseStrategy):
    """A strategy for the CLIN Agent that uses a general approach.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating responses.
        memory (CLINMemory): An instance of a memory used for storing and retrieving information.
        max_trials (int): The maximum number of trials allowed.
        max_steps (int): The maximum number of steps allowed.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[CLINMemory] = None,
        max_trials: int = 3,
        max_steps: int = 6,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        memory = memory or CLINMemory()

        super().__init__(
            llm=llm,
            memory=memory,
            max_trials=max_trials,
            max_steps=max_steps,
            testing=testing,
        )

    def generate(
        self,
        question: str,
        key: str,
        examples: str,
        prompt: str,
        summary_prompt: str,
        meta_summary_prompt: str,
        additional_keys: Dict[str, str],
        summary_additional_keys: Dict[str, str],
        meta_summary_additional_keys: Dict[str, str],
        summary_system: str,
        meta_summary_system: str,
        quadrant: str,
        patience: int,
        reset: bool,
    ) -> CLINOutput:
        """Generates an answer.

        Args:
            question (str): The question to be answered.
            key (str): The key used for storing and retrieving information.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            summary_prompt (str): The instruction template used to prompt the language model for the summary.
            meta_summary_prompt (str): The instruction template used to prompt the language model for the meta-summary.
            additional_keys (Dict[str, str]): Additional keys to format the answer and critique prompts.
            summary_additional_keys (Dict[str, str]): Additional keys to format the summary prompt.
            meta_summary_additional_keys (Dict[str, str]): Additional keys to format the meta-summary prompt.
            summary_system (str): The system message for the summary.
            meta_summary_system (str): The system message for the meta-summary.
            quadrant (str): The quadrant for the agent.
            patience (int): The patience for the agent.
            reset (bool): Whether to reset the agent.

        Returns:
            CLINOutput: The generated answer and critique.
        """
        start = time.time()

        # Reset.
        if reset:
            self.reset()

        scratchpad = ""
        answer = ""
        finished = False
        idx, step_idx, patience_cnt = 1, 1, 0
        steps: List[CLINStepOutput] = []

        # Load meta-summaries if applicable.
        if quadrant == "gen_env" or quadrant == "gen_task":
            meta_summaries = self.memory.load_meta_summaries()["meta_summaries"]
        else:
            meta_summaries = ""

        while not self.halting_condition(idx=idx, key=key, answer=answer):
            # Load previous memories.
            previous_memories = self.memory.load_memories(question=question)
            summaries = previous_memories["latest_summaries"]
            previous_trials = previous_memories["previous_trials"]

            # Generate ReAct trial.
            step_idx, is_correct, scratchpad, finished, answer, react_steps = (
                self.generate_react(
                    question=question,
                    key=key,
                    examples=examples,
                    summaries=summaries,
                    summary_system=summary_system,
                    meta_summaries=meta_summaries,
                    meta_summary_system=meta_summary_system,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )

            # Generate summaries.
            summaries, summaries_response = self.generate_summary(
                question=question,
                previous_trials=previous_trials,
                scratchpad=scratchpad,
                is_correct=is_correct,
                prompt=summary_prompt,
                additional_keys=summary_additional_keys,
            )

            steps.append(
                CLINStepOutput(
                    steps=react_steps,
                    summaries=summaries,
                    summaries_response=summaries_response,
                    meta_summaries=meta_summaries,
                    previous_trials=previous_trials,
                )
            )

            # Increment patience counter.
            if not is_correct:
                patience_cnt += 1
            if patience_cnt == patience:
                break

            idx += 1

        # Generate meta-summary.
        meta_summaries_response = None
        if quadrant == "gen_env" or quadrant == "gen_task":
            meta_summaries, meta_summaries_response = self.generate_meta_summary(
                question=question,
                meta_summaries=meta_summaries,
                meta_summary_system=meta_summary_system,
                previous_trials=previous_trials,
                scratchpad=scratchpad,
                prompt=meta_summary_prompt,
                additional_keys=meta_summary_additional_keys,
            )

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps, meta_summaries_response)
        out = CLINOutput(
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

    def generate_react(
        self,
        question: str,
        key: str,
        examples: str,
        summaries: str,
        summary_system: str,
        meta_summaries: str,
        meta_summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[int, bool, str, bool, str, List[CLINReActStepOutput]]:
        """Generates a reaction based on the given question, key, examples, reflections, prompt, and additional keys.

        Args:
            question (str): The question to be answered.
            key (str): The key for the observation.
            examples (str): Examples to guide the reaction process.
            summaries (str): The summaries of the previous steps.
            summary_system (str): The system prompt for the summaries.
            meta_summaries (str): The meta-summaries of the previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            prompt (str): The prompt or instruction to guide the reaction.
            additional_keys (Dict[str, str]): Additional keys for the reaction process.

        Returns:
            Tuple[int, bool, str, bool, str, List[CLINReActStepOutput]]: The reaction, whether the reaction is finished, the answer, whether the reaction is valid, the scratchpad, and the steps.
        """
        react_steps = []
        step_idx = 1
        scratchpad = ""
        finished = False
        answer = ""
        while not self.react_halting_condition(
            finished=finished,
            idx=step_idx,
        ):
            # Think.
            scratchpad, thought, thought_response = self.generate_thought(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                summaries=summaries,
                summary_system=summary_system,
                meta_summaries=meta_summaries,
                meta_summary_system=meta_summary_system,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_response = self.generate_action(
                idx=step_idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                summaries=summaries,
                summary_system=summary_system,
                meta_summaries=meta_summaries,
                meta_summary_system=meta_summary_system,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, finished, is_correct, obs, external_tool_info = (
                self.generate_observation(
                    idx=step_idx,
                    scratchpad=scratchpad,
                    action_type=action_type,
                    query=query,
                    key=key,
                )
            )

            react_steps.append(
                CLINReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    is_correct=is_correct,
                    thought_response=thought_response,
                    action_response=action_response,
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
        summaries: str,
        summary_system: str,
        meta_summaries: str,
        meta_summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generates a thought based on the given question, examples, summaries, prompt, and additional keys.

        Args:
            idx (int): The current step.
            scratchpad (str): The scratchpad containing previous thoughts.
            question (str): The question to generate a thought for.
            examples (str): Examples to guide the thought generation process.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for the summaries.
            meta_summaries (str): Meta-summaries of previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            prompt (str): The prompt or instruction to guide the thought generation.
            additional_keys (Dict[str, str]): Additional keys for the thought generation process.

        Returns:
            Tuple[str, str, Response]: The updated scratchpad, the generated thought, and the thought responses.
        """
        scratchpad += f"\nThought {idx}: "
        out = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            summaries=summaries,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            summary_system=summary_system,
            meta_summaries=meta_summaries,
            meta_summary_system=meta_summary_system,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(out.output_text).split("Action")[0].strip()
        scratchpad += thought

        return scratchpad, thought, out

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        summaries: str,
        summary_system: str,
        meta_summaries: str,
        meta_summary_system: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action for the current step in the reasoning process.

        Args:
            idx (int): The current step index.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The main question or task to be addressed.
            examples (str): Relevant examples to provide context for action generation.
            trajectory (str): The current trajectory or history of thoughts and actions.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for the summaries.
            meta_summaries (str): Meta-summaries of previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, Response]: A tuple containing the updated trajectory, action type, query, and the metrics.
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
                - A boolean indicating if the task is finished.
                - The generated observation.
                - The observation.
                - A dictionary with additional information.
        """
        raise NotImplementedError

    def generate_summary(
        self,
        question: str,
        previous_trials: str,
        scratchpad: str,
        is_correct: bool,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generates a summary based on the given inputs.

        Args:
            question (str): The question to be answered.
            previous_trials (str): The previous trials.
            scratchpad (str): The scratchpad containing previous thoughts.
            is_correct (bool): Whether the answer is correct.
            prompt (str): The prompt or instruction to guide the summary generation.
            additional_keys (Dict[str, str]): Additional keys for the summary generation.

        Returns:
            Tuple[str, Response]: The generated summary or response.
        """
        raise NotImplementedError

    def generate_meta_summary(
        self,
        question: str,
        meta_summaries: str,
        meta_summary_system: str,
        previous_trials: str,
        scratchpad: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generates a meta-summary based on the given inputs.

        Args:
            question (str): The question to be answered.
            meta_summaries (str): The meta-summaries of the previous steps.
            meta_summary_system (str): The system prompt for the meta-summaries.
            previous_trials (str): The previous trials.
            scratchpad (str): The scratchpad containing previous thoughts.
            prompt (str): The prompt or instruction to guide the meta-summary generation.
            additional_keys (Dict[str, str]): Additional keys for the meta-summary generation.

        Returns:
            Tuple[str, Response]: The generated meta-summary.
        """
        out = _prompt_meta_summary(
            llm=self.llm,
            question=question,
            meta_summary_system=meta_summary_system,
            meta_summaries=meta_summaries,
            previous_trials=previous_trials,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        # Add meta-summaries to memory.
        self.memory.add_meta_summaries(
            question=question,
            meta_summaries=out.output_text,
        )

        return out.output_text, out

    def halting_condition(
        self,
        idx: int,
        key: str,
        answer: str,
    ) -> bool:
        """Determine whether the halting condition has been met in the CLIN agent.

        Args:
            idx (int): The index of the current step.
            key (str): The key for the observation.
            answer (str): The answer to the question.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    def react_halting_condition(
        self,
        finished: bool,
        idx: int,
    ) -> bool:
        """Determine whether the halting condition has been met in the ReflexionReAct agent.

        Args:
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            question (str): The question to generate an action for.
            examples (str): Examples to guide the action generation process.
            summaries (str): Summaries of previous steps.
            summary_system (str): The system prompt for summarization.
            meta_summaries (str): Meta-summaries of previous steps.
            meta_summary_system (str): The system prompt for meta-summarization.
            prompt (str): The prompt or instruction to guide the action generation.
            additional_keys (Dict[str, str]): Additional keys for the action generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise. The halting condition is met when the answer is not correct and the current step index is less than the maximum number of steps plus one.
        """
        return _is_halted(
            finished=finished,
            step_idx=idx,
            max_steps=self.max_steps,
        )

    def reset(self) -> None:
        """Resets the strategy's internal state."""
        self.memory.clear()
