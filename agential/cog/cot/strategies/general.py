"""CoT general strategy."""

import time
from typing import Dict
from agential.cog.cot.output import CoTOutput
from agential.cog.cot.strategies.base import CoTBaseStrategy
from agential.cog.cot.functional import _prompt_agent
from agential.llm.llm import BaseLLM

class CoTGeneralStrategy(CoTBaseStrategy):
    """The general strategy for the CoT Agent.

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

        out = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            prompt=prompt,
            additional_keys=additional_keys,
        )

        total_time = time.time() - start
        return CoTOutput(
            answer=out.output_text.split("Finish[")[-1].split("]")[0],
            total_prompt_tokens=out.prompt_tokens,
            total_completion_tokens=out.completion_tokens,
            total_tokens=out.total_tokens,
            total_prompt_cost=out.prompt_cost,
            total_completion_cost=out.completion_cost,
            total_cost=out.total_cost,
            total_prompt_time=out.prompt_time,
            total_time=total_time if not self.testing else 0.5,
            additional_info=out
        )
        
    def reset(self) -> None:
        """Resets the strategy's internal state."""
        pass