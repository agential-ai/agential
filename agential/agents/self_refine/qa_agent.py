"""Self-Refine QA Agent for question-answering benchmarks."""

from typing import Any, Dict, Optional
import time
from agential.core.llm import BaseLLM
from agential.agents.base import BaseMethod

from agential.eval.classification import EM, fuzzy_EM
from .utils import log_llm_io


class SelfRefineQA(BaseMethod):
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        patience: int = 1,
        testing: bool = False,
        max_interactions: int = 3,
        verbose: bool = False,
        config: dict = {},
        truncate_length: int = -1,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.patience = patience
        self.testing = testing
        self.max_interactions = max_interactions
        self._prev_answer = ""
        self.patience_counter = 0
        self.truncate_length = truncate_length

    def halting_condition(self, answer: str) -> bool:
        # Check both exact match and fuzzy match
        exact_match = EM(answer.strip(), self._prev_answer, normalize=False)
        fuzzy_match_result = fuzzy_EM(
            answer.strip(), self._prev_answer, normalize=False, fuzzy_threshold=0.95
        )

        if exact_match or fuzzy_match_result:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                return True
        else:
            self._prev_answer = answer.strip()
            self.patience_counter = 0
        return False

    def generate(
        self,
        question: str,
        fewshot_type: str = "",
        examples: Optional[str] = None,
        prompt: Optional[str] = None,
        critique_examples: Optional[str] = None,
        critique_prompt: Optional[str] = None,
        refine_examples: Optional[str] = None,
        refine_prompt: Optional[str] = None,
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        refine_additional_keys: Dict[str, str] = {},
        max_interactions: Optional[int] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        # Reset state for new generation
        self._prev_answer = ""
        self.patience_counter = 0
        steps = []
        scratchpad = ""
        total_tokens = total_cost = 0
        # Select fewshots and prompts if not provided
        if not (
            examples
            and prompt
            and critique_examples
            and critique_prompt
            and refine_examples
            and refine_prompt
        ):
            examples = examples or self.config.get("examples", "")
            critique_examples = critique_examples or self.config["critique_examples"]
            refine_examples = refine_examples or self.config["refine_examples"]
            prompt = prompt or self.config["prompt"]
            critique_prompt = critique_prompt or self.config["critique_prompt"]
            refine_prompt = refine_prompt or self.config["refine_prompt"]
        max_iters = max_interactions or self.max_interactions
        answer = ""
        for idx in range(1, max_iters + 1):
            # 1. Generate answer (or refinement)
            if idx == 1:
                input_prompt = prompt.format(
                    examples=examples,
                    question=question,
                    **additional_keys,
                )
                response = self.llm(input_prompt)
                log_llm_io(
                    response,
                    f"Step {idx} - Initial Answer",
                    self.verbose,
                    self.truncate_length,
                )
                answer = response.output_text.strip()
                step_tokens = response.total_tokens
                step_cost = response.total_cost
            else:
                input_prompt = refine_prompt.format(
                    examples=refine_examples,
                    question=question,
                    answer=answer,
                    critique=critique,
                    **refine_additional_keys,
                )
                response = self.llm(input_prompt)
                log_llm_io(
                    response, f"Step {idx} - Refine", self.verbose, self.truncate_length
                )
                answer = response.output_text.split("Answer: ")[-1].strip()
                step_tokens = response.total_tokens
                step_cost = response.total_cost
            scratchpad += f"\nAnswer {idx}: {answer}"
            # 2. Generate critique
            critique_input = critique_prompt.format(
                examples=critique_examples,
                question=question,
                answer=answer,
                **critique_additional_keys,
            )
            critique_response = self.llm(critique_input)
            log_llm_io(
                critique_response,
                f"Step {idx} - Critique",
                self.verbose,
                self.truncate_length,
            )
            critique = critique_response.output_text.strip()
            scratchpad += f"\nCritique {idx}: {critique}"
            step_tokens += critique_response.total_tokens
            step_cost += critique_response.total_cost
            total_tokens += step_tokens
            total_cost += step_cost
            steps.append(
                {
                    "answer": answer,
                    "critique": critique,
                    "answer_prompt": input_prompt,
                    "critique_prompt": critique_input,
                    "step_tokens": step_tokens,
                    "step_cost": step_cost,
                }
            )
            if self.halting_condition(answer):
                break
        total_time = time.time() - start_time
        return {
            "answer": answer,
            "steps": steps,
            "scratchpad": scratchpad,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
            },
        }
