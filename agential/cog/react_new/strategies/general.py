"""General strategy for the ReAct Agent."""

from typing import Any, Dict, Tuple, List
import re

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react_new.functional import _is_halted, _prompt_agent
from agential.cog.react_new.strategies.base import ReActBaseStrategy
from agential.cog.react_new.output import ReActStepOutput, ReActOutput
from agential.llm.llm import BaseLLM
from agential.utils.general import get_token_cost_time
from agential.utils.parse import remove_newline

def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct.

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
            thought = self.generate_thought(
                scratchpad=scratchpad,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            scratchpad += thought

            # Act.
            scratchpad += f"\nAction {idx}: "

            action_type, query = self.generate_action(
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
                scratchpad=scratchpad, idx=idx, action_type=action_type, query=query
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
                    prompt_metrics={},
                )
            )

            idx += 1

        return steps
    
    def generate_thought(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ):
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

        return thought

    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ):
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.choices[0].message.content

        action = remove_newline(action).split("Observation")[0]
        action_type, query = parse_qa_action(action)

        return action_type, query
    
    def generate_observation(
        self, idx: int, action_type: str, query: str
    ):
        answer = ""
        finished = False
        external_tool_info = {"search_result": "", "lookup_result": ""}

        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = query
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)

            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."

        return answer, obs, finished, external_tool_info
    
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

    def reset(self, **kwargs: Any) -> None:
        pass