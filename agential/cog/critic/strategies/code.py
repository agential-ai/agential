"""CRITIC Agent strategies for Code."""

from typing import Any, Dict, Tuple

from agential.cog.critic.functional import _prompt_agent, _prompt_critique
from agential.cog.critic.strategies.base import CriticBaseStrategy
from agential.llm.llm import BaseLLM
from agential.utils.general import safe_execute
from agential.utils.metrics import get_prompt_info
from agential.utils.validation import validate_overlapping_keys


class CriticCodeStrategy(CriticBaseStrategy):
    """A strategy class for Code benchmarks using the CRITIC agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
    """

    def __init__(self, llm: BaseLLM) -> None:
        """Initialization."""
        super().__init__(llm)
        self._halt = False
        self._prompt_metrics: Dict[str, Any] = {
            "answer": None,
            "critique": None,
            "updated_answer": None,
        }

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The math question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.
            **kwargs (Any): Additional arguments.

        Returns:
            str: The generated answer.
        """
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["answer"] = get_prompt_info(out)
        answer = out.choices[0].message.content
        answer = answer.split("```python")[-1].split("```")[0].strip("\n")

        return answer

    def generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_tool: bool,
        max_interactions: int,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates a critique for the provided answer using the given prompt and examples.

        This method does the following:
            1. Initializes an empty dictionary for external tool information.
            2. If `use_tool` is True:
                a. Checks if "tests" is in `additional_keys` and raises a ValueError if not.
                b. Executes the answer as code along with the provided tests.
                c. If the execution status is "Done", sets the `_halt` flag to True.
                d. Updates the external tool information with the execution status.
                e. Validates and merges additional keys with external tool information.
            3. Copies the additional keys and updates them with external tool information.
            4. Generates a new critique using the updated answer and keys.
            5. Returns the new critique and external tool information.

        Args:
            idx (int): The index of the current interaction.
            question (str): The math question that was answered.
            examples (str): Few-shot examples to guide the critique.
            answer (str): The answer to critique.
            critique (str): Existing critique to build upon.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.
            use_tool (bool): Whether to use an external tool during critique.
            max_interactions (int): The maximum number of interactions allowed.
            **kwargs (Any): Additional arguments for specific implementations.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated critique and external tool information.
        """
        external_tool_info = {"execution_status": ""}
        if use_tool:
            if "tests" not in additional_keys:
                raise ValueError(
                    "The 'tests' parameter must be specified in `critique_additional_keys`."
                )
            tests = additional_keys["tests"]

            _, execution_status = safe_execute(f"{answer}\n\n{tests}")
            if execution_status == "Done":
                self._halt = True
            external_tool_info = {
                "execution_status": execution_status,
            }

            validate_overlapping_keys(additional_keys, external_tool_info)

        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info if use_tool else {})

        out = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["critique"] = get_prompt_info(out)
        new_critique = out.choices[0].message.content
        new_critique = new_critique.split("Here's")[0]

        return new_critique, external_tool_info

    def create_output_dict(
        self, answer: str, critique: str, external_tool_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Creates an output dictionary containing the answer, critique, and external tool information.

        Args:
            answer (str): The generated answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, Any]): Information from external tool execution.

        Returns:
            Dict[str, Any]: The output dictionary with the answer, critique, and external tool info.
        """
        output_dict = {
            "answer": answer,
            "critique": critique,
            "external_tool_info": external_tool_info,
            "prompt_metrics": self._prompt_metrics,
        }
        return output_dict

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.
            external_tool_info: Information from any external tool used.
            **kwargs (Any): Additional parameters for flexibility.

        Returns:
            str: The updated answer.
        """
        validate_overlapping_keys(additional_keys, external_tool_info)
        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        out = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=f"{critique}\n\nHere's a better solution:\n```python\n",
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["updated_answer"] = get_prompt_info(out)
        new_answer = out.choices[0].message.content
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip()

        return new_answer

    def halting_condition(self) -> bool:
        """Checks if the halting condition has been met.

        Returns True if the CRITIC Agent's generated answer has an `execution_status="Done"`.

        Returns:
            bool: True if the halting condition has been met, False otherwise.
        """
        return self._halt

    def reset(self, **kwargs: Any) -> None:
        """Resets the strategy to its initial state.

        Resets internal variables keeping track of halting and answer history.

        Args:
            **kwargs (Any): Additional arguments.

        Returns:
            None
        """
        self._halt = False
        self._prompt_metrics = {
            "answer": None,
            "critique": None,
            "updated_answer": None,
        }


class CritMBPPCodeStrategy(CriticCodeStrategy):
    """A strategy class for the MBPP benchmark using the CRITIC agent."""

    pass


class CritHEvalCodeStrategy(CriticCodeStrategy):
    """A strategy class for the HumanEval benchmark using the CRITIC agent."""

    def generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_tool: bool,
        max_interactions: int,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates a critique for the provided answer using the given prompt and examples.

        This method does the following:
            1. Initializes an empty dictionary for external tool information.
            2. If `use_tool` is True:
                a. Checks if "tests" is in `additional_keys` and raises a ValueError if not.
                b. Executes the answer as code along with the provided tests.
                c. If the execution status is "Done", sets the `_halt` flag to True.
                d. Updates the external tool information with the execution status.
                e. Validates and merges additional keys with external tool information.
            3. Copies the additional keys and updates them with external tool information.
            4. Generates a new critique using the updated answer and keys.
            5. Returns the new critique and external tool information.

        Args:
            idx (int): The index of the current interaction.
            question (str): The math question that was answered.
            examples (str): Few-shot examples to guide the critique.
            answer (str): The answer to critique.
            critique (str): Existing critique to build upon.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.
            use_tool (bool): Whether to use an external tool during critique.
            max_interactions (int): The maximum number of interactions allowed.
            **kwargs (Any): Additional arguments for specific implementations.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated critique and external tool information.
        """
        external_tool_info = {}
        if use_tool:
            if "tests" not in additional_keys:
                raise ValueError(
                    "The 'tests' parameter must be specified in `critique_additional_keys`."
                )
            tests = additional_keys["tests"]

            _, execution_status = safe_execute(f"{question}{answer}\n\n{tests}")
            if execution_status == "Done":
                self._halt = True
            external_tool_info = {
                "execution_status": execution_status,
            }
            validate_overlapping_keys(additional_keys, external_tool_info)

        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        out = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["critique"] = get_prompt_info(out)
        new_critique = out.choices[0].message.content

        new_critique = (
            new_critique.split("Here's")[0]
            .split("Here is")[0]
            .split("```python")[0]
            .strip("\n")
        )

        return new_critique, external_tool_info

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.
            external_tool_info: Information from any external tool used.
            **kwargs (Any): Additional parameters for flexibility.

        Returns:
            str: The updated answer.
        """
        validate_overlapping_keys(additional_keys, external_tool_info)
        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        out = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=f"{critique}\n\nHere's a better solution (include only function implementation):\n```python\n{question}",
            prompt=prompt,
            additional_keys=additional_keys,
        )
        self._prompt_metrics["updated_answer"] = get_prompt_info(out)
        new_answer = out.choices[0].message.content
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip("\n")

        return new_answer
