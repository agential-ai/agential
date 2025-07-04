# StandardMathAgent: Math agent for standard prompting (native dicts, all logic in this file)

from agential.methods.base import BaseMethod
from .utils import log_llm_io
import time
from agential.utils.general import safe_execute
from agential.eval.classification import EM

def extract_code_block(text):
    # Extract code block from markdown, or return stripped text
    if "```" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            return code.strip()
        parts = text.split("```", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            return code.strip()
    return text.strip()

class StandardMath(BaseMethod):
    def __init__(self, llm, benchmark, verbose=False, config=None):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose, config=config or {})

    def generate(
        self,
        question: str,
        key: str = "",
        additional_keys: dict = {},
        prompt: str = "",
        fewshot: str = "",
        max_interactions: int = 1,
    ):
        start_time = time.time()
        prompt_template = prompt or self.config["prompt_template"]
        fewshot_examples = fewshot or self.config["fewshot"]
        steps = []
        done = False
        answer = None
        for _ in range(max(max_interactions, 1)):
            prompt_str = prompt_template.format(
                question=question,
                examples=fewshot_examples,
                **additional_keys
            )
            response = self.llm(prompt_str)
            log_llm_io(response, context="StandardMath", verbose=True)
            
            # Extract code from response (following the strategy pattern)
            answer_text = response.output_text.strip()
            # Extract code block following the strategy pattern
            if "```python" in answer_text:
                answer_text = answer_text.split("```python")[-1].split("```")[0].strip()
            else:
                answer_text = extract_code_block(answer_text)
            
            steps.append({
                "answer": answer_text,
                "response": response,
                "prompt": prompt_str,
            })
            
            # Check if we got the correct answer (following the strategy pattern)
            code_answer, _ = safe_execute(answer_text)
            if EM(str(code_answer), key, is_numeric=True):
                answer = answer_text
                break
        
        if answer is None:
            # Use last answer
            answer = steps[-1]["answer"] if steps else None
            
        total_prompt_tokens = sum(getattr(s["response"], "prompt_tokens", 0) for s in steps)
        total_completion_tokens = sum(getattr(s["response"], "completion_tokens", 0) for s in steps)
        total_tokens = sum(getattr(s["response"], "total_tokens", 0) for s in steps)
        total_cost = sum(getattr(s["response"], "total_cost", 0.0) for s in steps)
        total_time = time.time() - start_time
        return {
            "answer": answer,
            "prompt": prompt_template,
            "steps": steps,
            "metrics": {
                "total_time": total_time,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
            },
        } 