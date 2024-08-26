"""Self-Refine general strategy."""


from typing import Any, Dict
from agential.cog.self_refine.output import SelfRefineOutput
from agential.cog.self_refine.strategies.base import SelfRefineBaseStrategy
from agential.llm.llm import BaseLLM


class SelfRefineGeneralStrategy(SelfRefineBaseStrategy):
    """A general strategy class for the Self-Refine agent.

    Attributes:
        llm (BaseLLM): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts. Defaults to 1.
        testing (bool): Whether the strategy is being used for testing. Defaults to False.
    """

    def __init__(self, llm: BaseLLM, patience: int = 1, testing: bool = False) -> None:
        """Initialization."""
        super().__init__(llm=llm, patience=patience, testing=testing)

        self._prev_answer = ""
        self.patience_counter = 0

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        critique_examples: str,
        critique_prompt: str,
        refine_examples: str,
        refine_prompt: str,
        additional_keys: Dict[str, str],
        critique_additional_keys: Dict[str, str],
        refine_additional_keys: Dict[str, str],
        max_interactions: int,
        reset: bool,    
    ) -> SelfRefineOutput:
        """Generates a refined solution for a given question through an iterative self-refinement process.
        Args:
            question (str): The question or problem to solve.
            examples (str): Precedent examples to guide initial solution generation.
            prompt (str): Instructional prompt for initial solution generation.
            critique_examples (str): Precedent examples to guide critique generation.
            critique_prompt (str): Instructional prompt for critique generation.
            refine_examples (str): Precedent examples to guide solution refinement.
            refine_prompt (str): Instructional prompt for refining the solution.
            additional_keys (Dict[str, str]): Additional keys to format the prompt.
            critique_additional_keys (Dict[str, str]): Additional keys to format the critique_prompt.
            refine_additional_keys (Dict[str, str]): Additional keys to format the refine_prompt.
            fewshot_type (str): The type of few-shot examples to use.
            max_interactions (int): Maximum number of refinement iterations.
            reset (bool): Resets the agent's state.
        Returns:
            SelfRefineOutput: The agent's output.
        """
        if reset:
            self.reset()

        out = []

        # Initial answer generation.
        answer = self.strategy.generate(question, examples, prompt, additional_keys)

        for _ in range(max_interactions):
            # Generate critique.
            critique = self.strategy.generate_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
            )

            out.append(
                SelfRefineOutput(**self.strategy.create_output_dict(answer, critique))
            )

            if self.strategy.halting_condition():
                break

            # Improve answer based on critique.
            answer = self.strategy.update_answer_based_on_critique(
                question=question,
                examples=refine_examples,
                answer=answer,
                critique=critique,
                prompt=refine_prompt,
                additional_keys=refine_additional_keys,
            )

        return out
    
    def reset(self) -> None:
        """Resets the strategy to its initial state."""
        self._prev_code_answer = ""
        self.patience_counter = 0
