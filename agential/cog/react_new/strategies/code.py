"""ReAct Agent strategies for Code."""

from typing import Any, Dict, Tuple

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react_new.functional import _prompt_agent, parse_code_action
from agential.cog.react_new.strategies.general import ReActGeneralStrategy
from agential.llm.llm import BaseLLM
from agential.utils.general import get_token_cost_time, safe_execute


class ReActCodeStrategy(ReActGeneralStrategy):
    """A strategy class for Code benchmarks using the ReAct agent.

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
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str]: The generated action type and code.
        """
        max_steps = kwargs.get("max_steps", self.max_steps)
        self._scratchpad += "\nAction:"
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=max_steps,  # type: ignore
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["action"] = get_token_cost_time(out)
        action = out.choices[0].message.content

        action = action.split("Observation")[0].strip()

        action_type, query = parse_code_action(action)
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
        external_tool_info = {"execution_status": ""}

        self._scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status

            self._answer = query
            self._finished = True
            obs = f"\n```python\n{self._answer}\n```"
        elif action_type.lower() == "implement":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status

            self._answer = query
            obs = f"\n```python\n{self._answer}\n```\nExecution Status: {execution_status}"
        elif action_type.lower() == "test":
            obs = f"{self._answer}\n\n{query}"
            _, execution_status = safe_execute(obs)
            external_tool_info["execution_status"] = execution_status

            obs = f"\n```python\n{obs}\n```\nExecution Status: {execution_status}"
        else:
            obs = "Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
        self._scratchpad += obs

        return obs, external_tool_info


class ReActMBPPStrategy(ReActCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReAct agent."""

    pass


class ReActHEvalStrategy(ReActCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReAct agent."""

    pass
