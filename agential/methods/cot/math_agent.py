from typing import Any, Dict, Optional
import time
from agential.core.llm import BaseLLM
from agential.agents.base import BaseMethod
from agential.eval.classification import EM
from agential.utils.general import safe_execute
from agential.agents.cot.utils import log_llm_io

class CoTMath(BaseMethod):
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
        code = answer.strip()
        if "```python" in code:
            code = code.split("```python")[-1].split("```", 1)[0].strip()
        try:
            code_with_imports = f"from typing import *\n{code}"
            code_answer, execution_status = safe_execute(code_with_imports)
            current_answer = str(code_answer[0]) if code_answer else ""
        except:
            current_answer = ""
        if EM(current_answer, self._prev_answer, normalize=False, is_numeric=True):
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                return True
        else:
            self._prev_answer = current_answer
            self.patience_counter = 0
        return False

    def generate(
        self,
        question: str,
        key: str = "",
        examples: Optional[str] = None,
        prompt: Optional[str] = None,
        additional_keys: Dict[str, str] = {},
        max_interactions: Optional[int] = None,
        warming: Optional[list] = None,
        num_retries: int = 1,
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
        max_iters = max_interactions or self.max_interactions
        answer = ""
        for idx in range(1, max_iters + 1):
            # 1. Generate thought
            input_prompt = prompt.format(
                examples=examples,
                question=question,
                **additional_keys,
            ) + f"\nThought:"
            response = self.llm(input_prompt)
            log_llm_io(response, f"Step {idx} - Thought", self.verbose, self.truncate_length)
            thought = response.output_text.strip()
            # 2. Generate answer (code)
            answer_prompt = input_prompt + f" {thought}\nAnswer:"
            answer_response = self.llm(answer_prompt)
            log_llm_io(answer_response, f"Step {idx} - Answer", self.verbose, self.truncate_length)
            answer = answer_response.output_text.strip()
            if "```python" in answer:
                answer = answer.split("```python")[-1].split("```", 1)[0].strip()
            answer = f"\n```python\n{answer}\n```\n"
            step_tokens = response.total_tokens + answer_response.total_tokens
            step_cost = response.total_cost + answer_response.total_cost
            scratchpad += f"\nThought {idx}: {thought}\nAnswer {idx}: {answer}"
            total_tokens += step_tokens
            total_cost += step_cost
            steps.append({
                "thought": thought,
                "answer": answer,
                "thought_prompt": input_prompt,
                "answer_prompt": answer_prompt,
                "step_tokens": step_tokens,
                "step_cost": step_cost,
            })
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