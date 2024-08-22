"""ReAct Agent strategies for Code."""

from typing import Any, Dict, Tuple

import tiktoken

from tiktoken.core import Encoding

from agential.cog.react.functional import _prompt_agent, parse_math_action
from agential.cog.react.strategies.general import ReActGeneralStrategy
from agential.llm.llm import BaseLLM, Response
from agential.utils.general import safe_execute


class ReActMathStrategy(ReActGeneralStrategy):
    """A strategy class for Math benchmarks using the ReAct agent.

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

    def generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generates an action based on the provided inputs, including the question, examples, prompt, and additional keys.

        Args:
            idx (int): The index of the current action.
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples to be used in the prompt.
            prompt (str): The prompt to be used for generating the action.
            additional_keys (Dict[str, str]): Additional keys to be used in the prompt.

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
        action_type, query = parse_math_action(action)
        scratchpad += f"{action_type}[\n```python\n{query}\n```\n]"

        return scratchpad, action_type, query, out

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generates an observation based on the provided action type and query.

        Args:
            idx (int): The index of the current observation.
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action performed (e.g. "Calculate" or "Finish").
            query (str): The query to be executed.

        Returns:
            Tuple[str, str, str, bool, Dict[str, Any]]: The updated scratchpad, the answer, the observation, a flag indicating if the task is finished, and a dictionary with information about the code execution.
        """
        answer = ""
        finished = False
        external_tool_info = {"execution_status": "", "code_answer": ""}
        code_answer, execution_status = safe_execute(query)

        scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            answer = query
            finished = True
            obs = f"\n```python\n{answer}\n```"
        elif action_type.lower() == "calculate":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            answer = query
            obs = f"\n```python\n{answer}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = (
                "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
            )
        scratchpad += obs

        return scratchpad, answer, obs, finished, external_tool_info


class ReActGSM8KStrategy(ReActMathStrategy):
    """A strategy class for the GSM8K benchmark using the ReAct agent."""

    pass


class ReActSVAMPStrategy(ReActMathStrategy):
    """A strategy class for the SVAMP benchmark using the ReAct agent."""

    pass


class ReActTabMWPStrategy(ReActMathStrategy):
    """A strategy class for the TabMWP benchmark using the ReAct agent."""

    pass
