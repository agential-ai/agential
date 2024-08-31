"""Code strategies for CoT."""

import time

from typing import Dict

from agential.prompting.cot.functional import _prompt_llm
from agential.prompting.cot.output import CoTOutput, CoTStepOutput
from agential.prompting.cot.strategies.general import CoTGeneralStrategy


class CoTMBPPStrategy(CoTGeneralStrategy):
    """A strategy class for the MBPP benchmark using CoT."""

    pass


class CoTHEvalStrategy(CoTGeneralStrategy):
    """A strategy class for the HumanEval benchmark using CoT."""

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> CoTOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.

        Returns:
            CoTOutput: The output of the CoT strategy.
        """
        start = time.time()

        thought_response = _prompt_llm(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = thought_response.output_text.split("\n")[0].strip()

        answer_response = _prompt_llm(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=f"{prompt}{thought}\nAction: ",
            additional_keys=additional_keys,
        )
        answer = answer_response.output_text.split("```python")[-1].split("```")[0]
        answer = f"```python\n{answer}\n```"

        step = CoTStepOutput(
            thought=thought,
            answer=answer,
            thought_response=thought_response,
            answer_response=answer_response,
        )
        total_time = time.time() - start
        out = CoTOutput(
            answer=answer,
            total_prompt_tokens=thought_response.prompt_tokens
            + answer_response.prompt_tokens,
            total_completion_tokens=thought_response.completion_tokens
            + answer_response.completion_tokens,
            total_tokens=thought_response.total_tokens + answer_response.total_tokens,
            total_prompt_cost=thought_response.prompt_cost
            + answer_response.prompt_cost,
            total_completion_cost=thought_response.completion_cost
            + answer_response.completion_cost,
            total_cost=thought_response.total_cost + answer_response.total_cost,
            total_prompt_time=thought_response.prompt_time
            + answer_response.prompt_time,
            total_time=total_time if not self.testing else 0.5,
            additional_info=step,
        )

        return out
