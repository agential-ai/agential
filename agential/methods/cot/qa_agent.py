from typing import Any, Dict, Optional
import time
from agential.core.llm import BaseLLM
from agential.methods.base import BaseMethod
from agential.eval.classification import EM, fuzzy_EM
from agential.methods.cot.utils import log_llm_io


class CoTQA(BaseMethod):
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        patience: int = 1,
        max_interactions: int = 1,
        verbose: bool = False,
        config: dict = {},
        truncate_length: int = -1,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config)
        self.patience = patience
        self.max_interactions = max_interactions
        self._prev_answer = ""
        self.patience_counter = 0
        self.truncate_length = truncate_length

    def halting_condition(self, answer: str) -> bool:
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
        key: str = "",
        examples: Optional[str] = None,
        prompt: Optional[str] = None,
        additional_keys: Dict[str, str] = {},
    ) -> Dict[str, Any]:
        start_time = time.time()
        self._prev_answer = ""
        self.patience_counter = 0
        steps = []
        scratchpad = ""
        total_tokens = total_cost = 0
        if not (examples and prompt):
            examples = examples or self.config.get("examples", "")
            prompt = prompt or self.config["prompt"]
        answer = ""
        for idx in range(1, self.max_interactions + 1):
            # 1. Generate thought
            input_prompt = (
                prompt.format(
                    examples=examples,
                    question=question,
                    **additional_keys,
                )
                + f"\nThought:"
            )
            response = self.llm(input_prompt)
            log_llm_io(
                response, f"Step {idx} - Thought", self.verbose, self.truncate_length
            )
            thought = response.output_text.strip()
            # 2. Generate answer
            answer_prompt = input_prompt + f" {thought}\nAnswer:"
            answer_response = self.llm(answer_prompt)
            log_llm_io(
                answer_response,
                f"Step {idx} - Answer",
                self.verbose,
                self.truncate_length,
            )
            answer = answer_response.output_text.strip()
            answer = answer.split("Finish[")[-1].split("]")[0]
            step_tokens = response.total_tokens + answer_response.total_tokens
            step_cost = response.total_cost + answer_response.total_cost
            scratchpad += f"\nThought {idx}: {thought}\nAnswer {idx}: {answer}"
            total_tokens += step_tokens
            total_cost += step_cost
            steps.append(
                {
                    "thought": thought,
                    "answer": answer,
                    "thought_prompt": input_prompt,
                    "answer_prompt": answer_prompt,
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
