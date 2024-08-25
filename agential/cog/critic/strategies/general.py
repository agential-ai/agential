"""CRITIC general strategy."""

import time

from typing import Any, Dict, List, Tuple

from agential.cog.critic.functional import accumulate_metrics
from agential.cog.critic.output import CriticOutput, CriticStepOutput
from agential.cog.critic.strategies.base import CriticBaseStrategy
from agential.llm.llm import BaseLLM, Response


class CriticGeneralStrategy(CriticBaseStrategy):
    """A general strategy class for the CRITIC agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        testing (bool): Whether to run in testing mode. Defaults to False.
    """

    def __init__(
        self,
        llm: BaseLLM,
        testing: bool = False,
    ) -> None:
        """Initialization."""
        super().__init__(
            llm=llm,
            testing=testing,
        )

    def generate(
        self,
        question: str,
        examples: str,
        critique_examples: str,
        prompt: str,
        critique_prompt: str,
        additional_keys: Dict[str, str],
        critique_additional_keys: Dict[str, str],
        max_interactions: int,
        use_tool: bool,
        reset: bool,
    ) -> CriticOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            critique_examples (str): Few-shot examples to guide the language model in generating the critique.
            prompt (str): The instruction template used to prompt the language model for the answer.
            critique_prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the answer and critique prompts.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            max_interactions (int): The maximum number of interactions to perform.
            use_tool (bool): Whether to use a tool for generating the critique.
            reset (bool): Whether to reset the strategy.

        Returns:
            CriticOutput: The generated answer and critique.
        """
        start = time.time()

        if reset:
            self.reset()

        steps: List[CriticStepOutput] = []

        # Initial answer generation.
        answer, answer_response = self.generate_answer(
            question, examples, prompt, additional_keys
        )

        critique = ""
        for idx in range(max_interactions):
            critique, external_tool_info, finished, critique_response = (
                self.generate_critique(
                    idx=idx,
                    question=question,
                    examples=critique_examples,
                    answer=answer,
                    critique=critique,
                    prompt=critique_prompt,
                    additional_keys=critique_additional_keys,
                    use_tool=use_tool,
                    max_interactions=max_interactions,
                )
            )

            steps.append(
                CriticStepOutput(
                    **self.create_output_dict(
                        finished=finished,
                        answer=answer,
                        critique=critique,
                        external_tool_info=external_tool_info,
                        answer_response=answer_response,
                        critique_response=critique_response,
                    )
                )
            )

            if self.halting_condition(finished=finished):
                break

            # Update answer for the next iteration.
            answer, answer_response = self.update_answer_based_on_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                critique=critique,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
                external_tool_info=external_tool_info,
            )

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        out = CriticOutput(
            answer=steps[-1].answer,
            total_prompt_tokens=total_metrics["total_prompt_tokens"],
            total_completion_tokens=total_metrics["total_completion_tokens"],
            total_tokens=total_metrics["total_tokens"],
            total_prompt_cost=total_metrics["total_prompt_cost"],
            total_completion_cost=total_metrics["total_completion_cost"],
            total_cost=total_metrics["total_cost"],
            total_prompt_time=total_metrics["total_prompt_time"],
            total_time=total_time if not self.testing else 0.5,
            additional_info=steps,
        )

        return out

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
        raise NotImplementedError

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
        """Generates a critique of the provided answer using the given language model, question, examples, and prompt.

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
        raise NotImplementedError

    def create_output_dict(
        self,
        finished: bool,
        answer: str,
        critique: str,
        external_tool_info: Dict[str, Any],
        critique_response: List[Response],
    ) -> Dict[str, Any]:
        """Creates a dictionary containing the answer and critique, along with any additional key updates.

        Args:
            finished (bool): Whether the critique process has finished.
            answer (str): The original answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, Any]): Information from any external tools used during the critique.
            critique_response (List[Response]): The responses from the critique.

        Returns:
            Dict[str, Any]: A dictionary containing the answer, critique, and additional key updates.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def halting_condition(self, finished: bool) -> bool:
        """Checks if the halting condition is met.

        Args:
            finished (bool): Whether the interaction has finished.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the strategy's internal state."""
        raise NotImplementedError
