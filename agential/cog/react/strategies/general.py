"""General strategy for the ReAct Agent."""

import time

from typing import Any, Dict, Tuple

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react.functional import _is_halted, _prompt_agent, accumulate_metrics
from agential.cog.react.output import ReActOutput, ReActStepOutput
from agential.cog.react.strategies.base import ReActBaseStrategy
from agential.llm.llm import BaseLLM, Response
from agential.utils.parse import remove_newline


class ReActGeneralStrategy(ReActBaseStrategy):
    """A general strategy class using the ReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
        testing (bool): Whether the agent is in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(
            llm=llm,
            max_steps=max_steps,
            max_tokens=max_tokens,
            enc=enc,
            testing=testing,
        )

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reset: bool,
    ) -> ReActOutput:
        """Generate a ReAct output by iteratively thinking, acting, and observing.

        Args:
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the thought.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.
            reset (bool): Whether to reset the agent's state before generating.

        Returns:
            ReActOutput: The generated output, including the final answer, metrics, and step-by-step details.
        """
        start = time.time()

        if reset:
            self.reset()

        scratchpad = ""
        answer = ""
        finished = False
        idx = 1
        steps = []
        while not self.halting_condition(
            finished=finished,
            idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        ):
            # Think.
            scratchpad, thought, thought_response = self.generate_thought(
                idx=idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Act.
            scratchpad, action_type, query, action_response = self.generate_action(
                idx=idx,
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )

            # Observe.
            scratchpad, answer, obs, finished, external_tool_info = (
                self.generate_observation(
                    idx=idx, scratchpad=scratchpad, action_type=action_type, query=query
                )
            )

            steps.append(
                ReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    thought_response=thought_response,
                    action_response=action_response,
                )
            )

            idx += 1

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        out = ReActOutput(
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
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generate a thought based on the given inputs.

        Args:
            idx (int): The current index of the thought.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the thought.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            Tuple[str, str, Response]: The updated scratchpad, the generated thought, and the metrics for the thought.
        """
        scratchpad += f"\nThought {idx}: "

        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
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
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action based on the given inputs.

        Args:
            idx (int): The current index of the action.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question being answered.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the action.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            Tuple[str, str, str, Response]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
        """
        raise NotImplementedError

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generate an observation based on the given inputs.

        Args:
            idx (int): The current index of the observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed.
            query (str): The query or action to observe.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: A tuple containing:
                - The updated scratchpad.
                - The generated observation.
                - The observation type.
                - A boolean indicating if the task is finished.
                - A dictionary with additional information.
        """
        raise NotImplementedError

    def halting_condition(
        self,
        finished: bool,
        idx: int,
        question: str,
        scratchpad: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the current iteration of the task should be halted based on various conditions.

        Args:
            finished (bool): Whether the task has been completed.
            idx (int): The current index of the iteration.
            question (str): The question being answered.
            scratchpad (str): The current state of the scratchpad.
            examples (str): Examples provided for the task.
            prompt (str): The prompt used to generate the action.
            additional_keys (Dict[str, str]): Additional key-value pairs to pass to the language model.

        Returns:
            bool: True if the task should be halted, False otherwise.
        """
        return _is_halted(
            finished=finished,
            idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self) -> None:
        """Resets the internal state."""
        pass
