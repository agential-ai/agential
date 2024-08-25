"""Reflexion Agent strategies for Code."""

from typing import Any, Dict, Optional, Tuple

import tiktoken

from tiktoken.core import Encoding

from agential.cog.reflexion.functional import (
    _is_halted,
    _prompt_cot_agent,
    _prompt_react_agent,
    parse_math_code_action_cot,
    parse_math_code_action_react,
)
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.reflexion.strategies.general import (
    ReflexionCoTGeneralStrategy,
    ReflexionReActGeneralStrategy,
)
from agential.eval.em import EM
from agential.llm.llm import BaseLLM, Response
from agential.utils.general import safe_execute


class ReflexionCoTCodeStrategy(ReflexionCoTGeneralStrategy):
    """A strategy class for Code benchmarks using the ReflexionCoT agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionCoTReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 3.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: Optional[ReflexionCoTReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 3,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionCoTReflector(llm=llm, max_reflections=max_reflections)
        super().__init__(
            llm=llm,
            reflector=reflector,
            max_reflections=max_reflections,
            max_trials=max_trials,
            testing=testing,
        )

    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, str, Response]: The updated scratchpad, the generated action, the action type, and the responses for the action.
        """
        scratchpad += f"\nAction: "
        out = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.output_text
        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_code_action_cot(action)
        scratchpad += f" {action_type}[\n```python\n{query}\n```\n]"

        return scratchpad, action_type, f"\n```python\n{query}\n```\n", out

    def generate_observation(
        self, scratchpad: str, action_type: str, query: str, key: str
    ) -> Tuple[str, str, bool, str]:
        """Generates an observation based on the action type and query.

        Args:
            scratchpad (str): The current state of the scratchpad.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.
            key (str): The key for the observation.

        Returns:
            Tuple[str, str, bool, str]: The updated scratchpad, the answer, a boolean indicating if the observation is correct, and the observation itself.
        """
        answer = ""
        query = query.split("```python")[-1].split("```")[0].strip()
        _, execution_status = safe_execute(f"{query}\n\n{key}")

        scratchpad += f"\nObservation: "
        if action_type.lower() == "finish":
            answer = query
            if EM(execution_status, "Done", normalize=False):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        else:
            obs = "Invalid action type, please try again. Valid action is Finish[```python<code>```]"

        scratchpad += obs

        return (
            scratchpad,
            f"\n```python\n{answer}\n```\n",
            EM(execution_status, "Done", normalize=False),
            obs,
        )

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
        _, execution_status = safe_execute(f"{answer}\n\n{key}")
        return EM(execution_status, "Done", normalize=False) or idx >= self.max_trials

    def reflect_condition(
        self,
        idx: int,
        reflect_strategy: Optional[str],
        key: str,
        answer: str,
    ) -> bool:
        """Determines whether the reflection condition has been met.

        Args:
            idx (int): The current step.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            key (str): The key for the observation.
            answer (str): The answer generated.

        Returns:
            bool: True if the reflection condition is met, False otherwise.
        """
        answer = answer.split("```python")[-1].split("```")[0].strip()
        _, execution_status = safe_execute(f"{answer}\n\n{key}")
        return (
            idx > 0
            and not EM(execution_status, "Done", normalize=False)
            and reflect_strategy is not None
        )


class ReflexionReActCodeStrategy(ReflexionReActGeneralStrategy):
    """A strategy class for Code benchmarks using the ReflexionReAct agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        reflector (Optional[ReflexionReActReflector]): The reflector used for generating reflections. Defaults to None.
        max_reflections (int): The maximum number of reflections allowed. Defaults to 3.
        max_trials (int): The maximum number of trials allowed. Defaults to 3.
        max_steps (int): The maximum number of steps allowed. Defaults to 6.
        max_tokens (int): The maximum number of tokens allowed. Defaults to 5000.
        enc (Encoding): The encoding for tokenization. Defaults to gpt-3.5-turbo.
        testing (bool): Whether the strategy is in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        reflector: Optional[ReflexionReActReflector] = None,
        max_reflections: int = 3,
        max_trials: int = 3,
        max_steps: int = 6,
        max_tokens: int = 5000,
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
        testing: bool = False,
    ) -> None:
        """Initialization."""
        if reflector is None:
            reflector = ReflexionReActReflector(
                llm=llm, max_reflections=max_reflections
            )
        super().__init__(
            llm=llm,
            reflector=reflector,
            max_reflections=max_reflections,
            max_trials=max_trials,
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
        reflections: str,
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
            reflections (str): Previous reflections to guide the action generation.
            depth (int): The current depth in the search tree.
            prompt (str): The prompt template for action generation.
            additional_keys (Dict[str, str]): Additional keys for prompt formatting.

        Returns:
            Tuple[str, str, str, Response]: A tuple containing the updated trajectory, action type, query, and the responses.
        """
        scratchpad += f"\nAction {idx}: "
        out = _prompt_react_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            reflections=reflections,
            scratchpad=scratchpad,
            max_steps=self.max_steps,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.output_text
        action = action.split("Observation")[0].strip()
        action_type, query = parse_math_code_action_react(
            action, ["Finish", "Test", "Implement"]
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
        query = query.split("```python")[-1].split("```")[0].strip()
        external_tool_info = {"execution_status": ""}

        answer = ""
        finished = False
        scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            obs = f"{query}\n\n{key}"
            _, execution_status = safe_execute(obs)
            external_tool_info["execution_status"] = execution_status
            self._answer = query
            answer = query
            finished = True

            if EM(execution_status, "Done", normalize=False):
                obs = "Answer is CORRECT"
            else:
                obs = "Answer is INCORRECT"
        elif action_type.lower() == "implement":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status
            execution_status = (
                ""  # Execution status may be done, but not necessarily correct.
            )
            self._answer = query
            answer = query
            obs = f"\n```python\n{answer}\n```\nExecution Status: {execution_status}"
        elif action_type.lower() == "test":
            obs = f"{self._answer}\n\n{query}"
            _, execution_status = safe_execute(obs)
            external_tool_info["execution_status"] = execution_status

            obs = f"\n```python\n{obs}\n```\nExecution Status: {execution_status}"
        else:
            execution_status = ""
            obs = "Invalid Action. Valid Actions are Implement[\n```python\n<code>\n```\n], Test[\n```python\n<code>\n```\n], and Finish[\n```python\n<answer>\n```\n]."
        scratchpad += obs

        return (
            scratchpad,
            f"\n```python\n{answer}\n```\n",
            finished,
            EM(execution_status, "Done", normalize=False),
            obs,
            external_tool_info,
        )

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

        _, execution_status = safe_execute(f"{answer}\n\n{key}")
        return (
            EM(execution_status, "Done", normalize=False) or idx >= self.max_trials + 1
        )

    def reflect_condition(
        self,
        answer: str,
        finished: bool,
        idx: int,
        scratchpad: str,
        reflect_strategy: Optional[str],
        question: str,
        examples: str,
        key: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determine whether the reflection condition has been met in the ReflexionReAct agent.

        Args:
            answer (str): The answer generated.
            finished (bool): A boolean indicating whether the task is finished.
            idx (int): The index of the current step.
            scratchpad (str): The scratchpad containing previous thoughts and actions.
            reflect_strategy (Optional[str]): The strategy to use for reflection.
            question (str): The question to be reflected upon.
            examples (str): Examples to guide the reflection process.
            key (str): The key for the observation.
            prompt (str): The prompt or instruction to guide the reflection.
            additional_keys (Dict[str, str]): Additional keys for the reflection process.

        Returns:
            bool: True if the reflection condition is met, False otherwise. The reflection condition is met when the agent is halted, the answer is not correct, and the reflection strategy is provided.
        """
        answer = answer.split("```python")[-1].split("```")[0].strip()

        halted = _is_halted(
            finished=finished,
            step_idx=idx,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            reflections=self.reflector.reflections_str,
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        _, execution_status = safe_execute(f"{answer}\n\n{key}")

        return (
            halted
            and not EM(execution_status, "Done", normalize=False)
            and reflect_strategy is not None
        )

    def reset(self) -> None:
        """Resets the internal state of the strategy."""
        self.reflector.reset()
        self._answer = ""


class ReflexionCoTHEvalStrategy(ReflexionCoTCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReflexionCoT agent."""

    def generate_action(
        self,
        scratchpad: str,
        question: str,
        examples: str,
        reflections: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            scratchpad (str): The current state of the scratchpad.
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            reflections (str): Reflections to consider during generation.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str, str, Response]: The updated scratchpad, the generated action, the action type, and the responses for the action.
        """
        scratchpad += f"\nAction: "
        out = _prompt_cot_agent(
            llm=self.llm,
            examples=examples,
            reflections=reflections,
            question=question,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = out.output_text
        action = action.split("Observation")[0].strip()
        query = action.split("```python")[-1].split("```")[0]
        action_type = "Finish"
        scratchpad += f"{action_type}[\n```python\n{query}\n```\n]"

        return scratchpad, action_type, f"\n```python\n{query}\n```\n", out


class ReflexionCoTMBPPStrategy(ReflexionCoTCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReflexionCoT agent."""

    pass


class ReflexionReActHEvalStrategy(ReflexionReActCodeStrategy):
    """A strategy class for the HumanEval benchmark using the ReflexionReAct agent."""

    pass


class ReflexionReActMBPPStrategy(ReflexionReActCodeStrategy):
    """A strategy class for the MBPP benchmark using the ReflexionReAct agent."""

    pass
