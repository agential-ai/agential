"""ReAct Agent strategies for Code."""

import re

from typing import Any, Dict, Tuple

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react.functional import _is_halted, _prompt_agent
from agential.cog.react.strategies.base import ReActBaseStrategy
from agential.llm.llm import BaseLLM
from agential.utils.general import get_token_cost_time, safe_execute
from agential.utils.parse import remove_newline


def parse_math_action(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`, `Calculate`) and extracts the
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
    match = re.search(r"\b(Finish|Calculate)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


class ReActMathStrategy(ReActBaseStrategy):
    """A strategy class for Math benchmarks using the ReAct agent.

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

    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
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

        action = action.split("Observation")[0].strip()

        action_type, query = parse_math_action(action)
        self._scratchpad += f" {action_type}[\n```python\n{query}\n```\n]"

        return action_type, query

    def generate_observation(
        self, idx: int, action_type: str, query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated observation and external tool outputs.
        """
        external_tool_info = {"execution_status": "", "code_answer": ""}
        code_answer, execution_status = safe_execute(query)

        self._scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            self._answer = query
            self._finished = True
            obs = f"\n```python\n{self._answer}\n```"
        elif action_type.lower() == "calculate":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            self._answer = query
            obs = f"\n```python\n{self._answer}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = (
                "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
            )
        self._scratchpad += obs

        return obs, external_tool_info
    

class ReActGSM8KStrategy(ReActMathStrategy):
    """A strategy class for the GSM8K benchmark using the ReAct agent."""

    pass


class ReActSVAMPStrategy(ReActMathStrategy):
    """A strategy class for the SVAMP benchmark using the ReAct agent."""

    pass


class ReActTabMWPStrategy(ReActMathStrategy):
    """A strategy class for the TabMWP benchmark using the ReAct agent."""

    pass
