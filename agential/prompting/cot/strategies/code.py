"""Code strategies for CoT."""

import time

from typing import Dict, List, Optional

from agential.prompting.cot.functional import _prompt_llm, accumulate_metrics
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
        num_retries: int,
        warming: List[Optional[float]],
    ) -> CoTOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.
            num_retries (int): Number of retries.
            warming (List[Optional[float]]): List of warmup temperatures.

        Returns:
            CoTOutput: The output of the CoT strategy.
        """
        start = time.time()

        steps: List[List[CoTStepOutput]] = []
        for _ in range(max(num_retries, 1)):
            warming_steps: List[CoTStepOutput] = []
            for temperature in warming:
                thought_response = _prompt_llm(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    prompt=prompt,
                    additional_keys=additional_keys,
                    temperature=temperature,
                )
                thought = thought_response.output_text.split("\n")[0].strip()

                answer_response = _prompt_llm(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    prompt=f"{prompt}{thought}\nAction: ",
                    additional_keys=additional_keys,
                    temperature=temperature,
                )
                answer = answer_response.output_text.split("```python")[-1].split("```")[0]
                answer = f"```python\n{answer}\n```"

                step = CoTStepOutput(
                    thought=thought,
                    answer=answer,
                    thought_response=thought_response,
                    answer_response=answer_response,
                )
                warming_steps.append(step)
            steps.append(warming_steps)

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        out = CoTOutput(
            answer=[[warm_step.answer for warm_step in step] for step in steps],
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