"""CRITIC Agent strategies for Math."""

from typing import Any, Dict, List, Tuple

from agential.agents.critic.functional import _prompt_agent, _prompt_critique
from agential.agents.critic.strategies.general import CriticGeneralStrategy
from agential.llm.llm import BaseLLM, Response
from agential.utils.general import safe_execute
from agential.utils.validation import validate_overlapping_keys


class CriticMathStrategy(CriticGeneralStrategy):
    """A strategy class for Math benchmarks using the CRITIC agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts. Defaults to 2.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(self, llm: BaseLLM, patience: int = 2, testing: bool = False) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)
        self.patience = patience
        self._answer_history: List[Dict[str, Any]] = []
        self._prev_code_answer = ""
        self.patience_counter = 0

    def generate_answer(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, List[Response]]:
        """Generates an answer to the given question using the provided examples and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.

        Returns:
            Tuple[str, List[Response]]: The generated answer and model responses.
        """
        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        answer = out.output_text
        answer = answer.split("```python")[-1].split("```")[0].strip()

        return answer, [out]

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
    ) -> Tuple[str, Dict[str, Any], bool, List[Response]]:
        """Generates a critique for the provided answer using the given prompt and examples.

        This method does the following:
            1. If `use_tool` is True, execute the answer as code and store the result.
            2. Update the answer history with the current answer and external tool info.
            3. Check if the current code answer is the same as the previous one:
               - If yes, increment the patience counter.
               - If the patience counter reaches the patience limit, set the halt flag.
               - Otherwise, update the previous code answer.
            4. Find the last valid answer from the history that includes external tool info.
            5. Validate and merge additional keys with external tool info.
            6. Generate a new critique using the updated answer and keys.
            7. Return the new critique and external tool info.

        Args:
            idx (int): The index of the current interaction.
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            critique (str): The previous critique, if any.
            prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            use_tool (bool): Whether to use an external tool for generating the critique.
            max_interactions (int): The maximum number of interactions to perform.

        Returns:
            Tuple[str, Dict[str, Any], bool, List[Response]]: The generated critique, any external tool information, a boolean for if it finished, and the responses.
        """
        external_tool_info = {"execution_status": "", "code_answer": ""}

        finished = False
        if use_tool:
            code_answer, execution_status = safe_execute(answer)
            external_tool_info = {
                "execution_status": execution_status,
                "code_answer": code_answer[0] if code_answer[0] is not None else "",
            }
            self._answer_history.append(
                {"answer": answer, "external_tool_info": external_tool_info}
            )

            if code_answer[0] == self._prev_code_answer:
                self.patience_counter += 1
                if self.patience_counter == self.patience:
                    finished = True
            else:
                self._prev_code_answer = code_answer[0]

            last_valid_idx = -1
            for i in range(len(self._answer_history) - 1, -1, -1):
                if (
                    self._answer_history[i]["external_tool_info"]["code_answer"]
                    is not None
                ):  # type: ignore
                    last_valid_idx = i
                    break

            external_tool_info = self._answer_history[last_valid_idx][
                "external_tool_info"
            ]  # type: ignore
            answer = self._answer_history[last_valid_idx]["answer"]  # type: ignore

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
        new_critique = out.output_text
        new_critique = new_critique.split("Here's")[0]

        return new_critique, external_tool_info, finished, [out]

    def create_output_dict(
        self,
        finished: bool,
        answer: str,
        critique: str,
        external_tool_info: Dict[str, Any],
        answer_response: List[Response],
        critique_response: List[Response],
    ) -> Dict[str, Any]:
        """Creates a dictionary containing the answer and critique, along with any additional key updates.

        Args:
            finished (bool): Whether the critique process has finished.
            answer (str): The original answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, Any]): Information from any external tools used during the critique.
            answer_response (List[Response]): The responses from the answer.
            critique_response (List[Response]): The responses from the critique.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, critique, and additional key updates.
        """
        output_dict = {
            "answer": answer,
            "critique": critique,
            "external_tool_info": external_tool_info,
            "critique_response": critique_response,
            "answer_response": answer_response,
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
    ) -> Tuple[str, List[Response]]:
        """Updates the answer based on the provided critique using the given language model and question.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the updated answer.
            answer (str): The original answer to be updated.
            critique (str): The critique of the original answer.
            prompt (str): The instruction template used to prompt the language model for the update.
            additional_keys (Dict[str, str]): Additional keys to format the update prompt.
            external_tool_info (Dict[str, str]): Information from any external tools used during the critique.

        Returns:
            str: The updated answer.
            List[Response]: The responses from the critique.
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
        new_answer = out.output_text
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip()

        return new_answer, [out]

    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        return finished

    def reset(self) -> None:
        """Resets the strategy to its initial state."""
        self._answer_history = []
        self._prev_code_answer = ""
        self.patience_counter = 0


class CriticGSM8KStrategy(CriticMathStrategy):
    """A strategy class for the GSM8K benchmark using the CRITIC agent."""

    pass


class CriticSVAMPStrategy(CriticMathStrategy):
    """A strategy class for the SVAMP benchmark using the CRITIC agent."""

    pass


class CriticTabMWPStrategy(CriticMathStrategy):
    """A strategy class for the TabMWP benchmark using the CRITIC agent."""

    pass
