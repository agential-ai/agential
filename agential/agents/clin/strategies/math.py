"""CLIN math strategy."""

from typing import Any, Dict, Optional, Tuple

from tiktoken import Encoding
from tiktoken.model import encoding_for_model as encoding_for_model

from agential.agents.clin.functional import (
    _prompt_react_agent,
    _prompt_summary,
    parse_math_code_action_react,
)
from agential.agents.clin.memory import CLINMemory
from agential.agents.clin.strategies.general import CLINGeneralStrategy
from agential.core.llm import BaseLLM, Response
from agential.eval.metrics.classification import EM
from agential.utils.general import safe_execute


class CLINMathStrategy(CLINGeneralStrategy):
    """A strategy for the CLIN Agent that uses a math approach.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating responses.
        memory (CLINMemory): An instance of a memory used for storing and retrieving information.
        max_trials (int): The maximum number of trials allowed.
        max_steps (int): The maximum number of steps allowed.
        max_tokens (int): The maximum number of tokens allowed.
        enc (Encoding): The encoding for tokenization.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        memory: Optional[CLINMemory] = None,
        max_trials: int = 3,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = ...,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        memory = memory or CLINMemory()
        super().__init__(
            llm=llm,
            memory=memory,
            max_trials=max_trials,
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
        scratchpad += f"\nAction {idx}: "
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
        action = out.output_text
        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_code_action_react(
            action, ["Finish", "Calculate"]
        )
        scratchpad += f"{action_type}[\n```python\n{query}\n```\n]"

        return scratchpad, action_type, f"\n```python\n{query}\n```\n", out

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
        external_tool_info = {"execution_status": "", "code_answer": ""}
        query = query.split("```python")[-1].split("```")[0].strip()
        code_answer, execution_status = safe_execute(query)

        answer = ""
        finished = False
        scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            answer = query
            finished = True

            if EM(str(code_answer[0]), key, is_numeric=True):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        elif action_type.lower() == "calculate":
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status

            answer = query
            obs = f"\n```python\n{answer}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = "Invalid Action. Valid Actions are Calculate[\\n```python\\n<code>\\n```\\n] and Finish[\\n```python\\n<answer>\\n```\\n]."
        scratchpad += obs

        return (
            scratchpad,
            f"\n```python\n{answer}\n```\n",
            finished,
            EM(str(code_answer[0]), key, is_numeric=True),
            obs,
            external_tool_info,
        )

    def generate_summary(
        self,
        question: str,
        previous_trials: str,
        scratchpad: str,
        is_correct: bool,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str | Response]:
        """Generates a summary based on the given inputs.

        Args:
            question (str): The question to be answered.
            previous_trials (str): The previous trials.
            scratchpad (str): The scratchpad containing previous thoughts.
            is_correct (bool): Whether the answer is correct.
            prompt (str): The prompt or instruction to guide the summary generation.
            additional_keys (Dict[str, str]): Additional keys for the summary generation.

        Returns:
            Tuple[str | Response]: The generated summary or response.
        """
        out = _prompt_summary(
            llm=self.llm,
            question=question,
            previous_trials=previous_trials,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        # Add summaries to memory.
        eval_report = "Answer is CORRECT" if is_correct else "Answer is INCORRECT"
        self.memory.add_memories(
            question=question,
            summaries=out.output_text,
            trial=f"Question: {question}\n{out.output_text}\nEVALUATION REPORT: {eval_report}",
            is_correct=is_correct,
        )

        return out.output_text, out

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
        answer = answer.split("```python")[-1].split("```")[0].strip()
        code_answer, _ = safe_execute(answer)
        return (
            EM(str(code_answer[0]), key, is_numeric=True) or idx >= self.max_trials + 1
        )


class CLINGSM8KStrategy(CLINMathStrategy):
    """A strategy class for the GSM8K benchmark using the CLIN agent."""

    pass


class CLINSVAMPStrategy(CLINMathStrategy):
    """A strategy class for the SVAMP benchmark using the CLIN agent."""

    pass


class CLINTabMWPStrategy(CLINMathStrategy):
    """A strategy class for the TabMWP benchmark using the CLIN agent."""

    def generate_summary(
        self,
        question: str,
        previous_trials: str,
        scratchpad: str,
        is_correct: bool,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str | Response]:
        """Generates a summary based on the given inputs.

        Args:
            question (str): The question to be answered.
            previous_trials (str): The previous trials.
            scratchpad (str): The scratchpad containing previous thoughts.
            is_correct (bool): Whether the answer is correct.
            prompt (str): The prompt or instruction to guide the summary generation.
            additional_keys (Dict[str, str]): Additional keys for the summary generation.

        Returns:
            Tuple[str | Response]: The generated summary or response.
        """
        out = _prompt_summary(
            llm=self.llm,
            question=question,
            previous_trials=previous_trials,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        # Add summaries to memory.
        eval_report = "Answer is CORRECT" if is_correct else "Answer is INCORRECT"
        self.memory.add_memories(
            question=question,
            summaries=out.output_text,
            trial=f"{question}\n{out.output_text}\nEVALUATION REPORT: {eval_report}",
            is_correct=is_correct,
        )

        return out.output_text, out