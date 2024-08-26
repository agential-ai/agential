"""Self-Refine general strategy."""


from typing import Dict, Tuple
from agential.cog.self_refine.output import SelfRefineOutput, SelfRefineStepOutput
from agential.cog.self_refine.strategies.base import SelfRefineBaseStrategy
from agential.llm.llm import BaseLLM, Response


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
        answer, answer_response = self.generate_answer(question, examples, prompt, additional_keys)

        for _ in range(max_interactions):
            # Generate critique.
            critique, finished, critique_response = self.generate_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
            )

            out.append(
                SelfRefineStepOutput(
                    answer=answer, 
                    critique=critique,
                    answer_response=answer_response,
                    critique_response=critique_response,
                )
            )

            if self.halting_condition(finished=finished):
                break

            # Improve answer based on critique.
            answer, answer_response = self.update_answer_based_on_critique(
                question=question,
                examples=refine_examples,
                answer=answer,
                critique=critique,
                prompt=refine_prompt,
                additional_keys=refine_additional_keys,
            )

        return out
    
    def generate_answer(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, Response]:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            Tuple[str, Response]: The generated answer and the response from the language model.
        """
        raise NotImplementedError

    def generate_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, bool, Response]:
        """Generates a critique for the provided answer using the given prompt and examples.

        Stops early if patience is reached and answer remains the same.

        Args:
            question (str): The qa question that was answered.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.

        Returns:
            Tuple[str, bool, Response]: The critique, a boolean indicating it's finished, and the model response.
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
    ) -> Tuple[str, Response]:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.

        Returns:
            Tuple[str, Response]: The updated answer and the model response.
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
        """Resets the strategy to its initial state."""
        raise NotImplementedError
