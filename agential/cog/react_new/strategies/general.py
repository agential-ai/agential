"""General strategy for the ReAct Agent."""

from typing import Any, Dict, Tuple
import time

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react_new.functional import _is_halted, _prompt_agent, accumulate_metrics
from agential.cog.react_new.strategies.base import ReActBaseStrategy
from agential.cog.react_new.output import ReActStepOutput, ReActOutput
from agential.llm.llm import BaseLLM, ModelResponse
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline

class ReActGeneralStrategy(ReActBaseStrategy):
    """A general strategy class using the ReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__(llm, max_steps, max_tokens, enc)

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        reset: bool,
    ) -> ReActOutput:
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
            scratchpad += f"\nThought {idx}: "
            thought, thought_model_response = self.generate_thought(
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            scratchpad += thought

            # Act.
            scratchpad += f"\nAction {idx}: "

            action_type, query, action_model_response = self.generate_action(
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            scratchpad += f"{action_type}[{query}]"

            # Observe.
            scratchpad += f"\nObservation {idx}: "
            answer, obs, finished, external_tool_info = self.generate_observation(
                action_type=action_type, query=query
            )
            scratchpad += obs

            steps.append(
                ReActStepOutput(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                    answer=answer,
                    external_tool_info=external_tool_info,
                    thought_metrics=get_token_cost_time(thought_model_response),
                    action_metrics=get_token_cost_time(action_model_response),
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
            total_time=total_time,
            additional_info=steps
        )

        return out
    
    def generate_thought(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, ModelResponse]:
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = out.choices[0].message.content

        thought = remove_newline(thought).split("Action")[0].strip()

        return thought, out

    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, ModelResponse]:
        
        raise NotImplementedError
    
    def generate_observation(
        self, action_type: str, query: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        
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
        pass