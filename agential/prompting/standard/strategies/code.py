"""Code strategies for standard prompting."""

import time

from typing import Dict, List, Optional

from agential.core.llm import BaseLLM
from agential.eval.metrics.classification import EM
from agential.prompting.standard.functional import _prompt_llm, accumulate_metrics
from agential.prompting.standard.output import StandardOutput, StandardStepOutput
from agential.prompting.standard.strategies.general import StandardGeneralStrategy
from agential.utils.general import safe_execute


class StandardCodeStrategy(StandardGeneralStrategy):
    """The Code strategy for the Standard prompting method.

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
        key: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        num_retries: int,
        warming: List[Optional[float]],
    ) -> StandardOutput:
        """Generates an answer and critique for the given question using the provided examples and prompts.

        Args:
            question (str): The question to be answered.
            key (str): The answer.
            examples (str): Few-shot examples to guide the language model in generating the answer.
            prompt (str): The instruction template used to prompt the language model for the answer.
            additional_keys (Dict[str, str]): Additional keys to format the answer prompt.
            num_retries (int): Number of retries.
            warming (List[Optional[float]]): List of warmup temperatures.

        Returns:
            StandardOutput: The output of the Standard strategy.
        """
        start = time.time()

        done = False
        steps: List[List[StandardStepOutput]] = []
        for _ in range(max(num_retries, 1)):
            warming_steps: List[StandardStepOutput] = []
            for temperature in warming:
                answer_response = _prompt_llm(
                    llm=self.llm,
                    question=question,
                    examples=examples,
                    prompt=prompt,
                    additional_keys=additional_keys,
                    temperature=temperature,
                )
                answer = answer_response.output_text.strip().split("```")[0]

                step = StandardStepOutput(
                    answer=f"\n```python\nfrom typing import *\n\n{answer}\n```\n",
                    answer_response=answer_response,
                )
                warming_steps.append(step)

                _, execution_status = safe_execute(
                    f"from typing import *\n\n{answer}\n{key}"
                )
                if EM(execution_status, "Done", normalize=False):
                    done = True
                    break

            steps.append(warming_steps)

            if done:
                break

        total_time = time.time() - start
        total_metrics = accumulate_metrics(steps)
        out = StandardOutput(
            answer=f"\n```python\nfrom typing import *\n\n{answer}\n```\n",
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


class StandardMBPPStrategy(StandardCodeStrategy):
    """A strategy class for the MBPP benchmark using standard vanilla prompting."""

    pass


class StandardHEvalStrategy(StandardCodeStrategy):
    """A strategy class for the HumanEval benchmark using standard vanilla prompting."""

    pass
