"""CoT general strategy."""

import time

from typing import Dict, List

from agential.llm.llm import BaseLLM
from agential.prompting.cot.functional import _prompt_llm, accumulate_metrics
from agential.prompting.cot.output import CoTOutput, CoTStepOutput
from agential.prompting.cot.strategies.base import CoTBaseStrategy


class CoTGeneralStrategy(CoTBaseStrategy):
    """The general strategy for the CoT.

    Attributes:
        llm (BaseLLM): An instance of a language model used for generating responses.
        testing (bool): Whether the generation is for testing purposes. Defaults to False.
    """

    def __init__(self, llm: BaseLLM, testing: bool = False) -> None:
        """Initialization."""
        super().__init__(llm=llm, testing=testing)

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        num_retries: int
    ) -> CoTOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.
            num_retries (int): Number of retries.
            
        Returns:
            CoTOutput: The output of the CoT strategy.
        """
        start = time.time()

        steps: List[CoTStepOutput] = []
        for _ in range(max(num_retries, 1)):
            thought_response = _prompt_llm(
                llm=self.llm,
                question=question,
                examples=examples,
                prompt=prompt,
                additional_keys=additional_keys,
            )
            thought = thought_response.output_text.split("Action")[0].strip()

            answer_response = _prompt_llm(
                llm=self.llm,
                question=question,
                examples=examples,
                prompt=f"{prompt}{thought}\nAction: ",
                additional_keys=additional_keys,
            )
            answer = answer_response.output_text.split("Finish[")[-1].split("]")[0]

            step = CoTStepOutput(
                thought=thought,
                answer=answer,
                thought_response=thought_response,
                answer_response=answer_response,
            )
            steps.append(step)

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        out = CoTOutput(
            answer=[step.answer for step in steps],
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

    def reset(self) -> None:
        """Resets the strategy's internal state."""
        pass
