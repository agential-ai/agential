"""ReAct Agent strategies for Code."""

from typing import Any, Dict, Tuple

import tiktoken
from tiktoken.core import Encoding

from agential.cog.react_new.functional import _prompt_agent, parse_math_action
from agential.cog.react_new.strategies.general import ReActGeneralStrategy
from agential.llm.llm import BaseLLM, ModelResponse
from agential.utils.general import safe_execute


class ReActMathStrategy(ReActGeneralStrategy):
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
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, ModelResponse]:
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
        action = out.choices[0].message.content
        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_action(action)
        scratchpad += f" {action_type}[\n```python\n{query}\n```\n]"

        return scratchpad, action_type, query, out

    def generate_observation(
        self, idx: int, scratchpad: str, action_type: str, query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
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
