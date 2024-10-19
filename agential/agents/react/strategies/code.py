"""ReAct Agent strategies for Code."""

from typing import Any, Dict, Tuple

import tiktoken

from tiktoken.core import Encoding

from agential.agents.react.functional import _prompt_agent, parse_code_action
from agential.agents.react.strategies.general import ReActGeneralStrategy
from agential.core.llm import BaseLLM, Response
from agential.utils.general import safe_execute


class ReActCodeStrategy(ReActGeneralStrategy):
    """A strategy class for Code benchmarks using the ReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        enc (Encoding): The encoding used for the language model.
        testing (bool): Whether the strategy is in testing mode. Defaults to False.
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

        self._answer = ""

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            idx (int): The index of the action.
            scratchpad (str): The scratchpad containing the previous steps.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, str, Response]: The updated scratchpad, the generated action, the action type, and the metrics for the action.
        """
        scratchpad += f"\nAction {idx}: "
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.output_text

        action = action.split("Observation")[0].strip()

        action_type, query = parse_code_action(action)
        scratchpad += f"{action_type}[\n```python\n{query}\n```\n]"

        return scratchpad, action_type, f"\n```python\n{query}\n```\n", out

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            scratchpad (str): The scratchpad containing the previous steps.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: The scratchpad, the answer, observation, whether the query is correct, and the observation metrics.
        """
        finished = False
        external_tool_info = {"execution_status": ""}
        query = query.split("```python")[-1].split("```")[0].strip()

        scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status

            self._answer = query
            finished = True
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
        scratchpad += obs

        return (
            scratchpad,
            f"\n```python\n{self._answer}\n```\n",
            obs,
            finished,
            external_tool_info,
        )

    def reset(self) -> None:
        """Resets internal state."""
        self._answer = ""


class ReActMBPPStrategy(ReActCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReAct agent."""

    pass


class ReActHEvalStrategy(ReActCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReAct agent."""

    pass
