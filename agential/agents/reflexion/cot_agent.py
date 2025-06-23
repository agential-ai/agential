"""ReflexionCoT Agent - Chain of Thought with Reflection."""

import time
import logging
from typing import List, Dict, Any, Tuple
from agential.agents.base import BaseAgent
from agential.core.llm import BaseLLM
from agential.agents.reflexion.prompts import (
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_INSTRUCTION_FEVER,
    REFLEXION_COT_INSTRUCTION_TRIVIAQA,
    REFLEXION_COT_INSTRUCTION_AMBIGNQ,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_COT_INSTRUCTION_SVAMP,
    REFLEXION_COT_INSTRUCTION_TABMWP,
    REFLEXION_COT_INSTRUCTION_HUMANEVAL,
    REFLEXION_COT_INSTRUCTION_MBPP,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
)

# Configuration for CoT benchmarks
COT_BENCHMARK_CONFIG = {
    "hotpotqa": {
        "prompt": REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        "fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "fever": {
        "prompt": REFLEXION_COT_INSTRUCTION_FEVER,
        "fewshot": FEVER_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "triviaqa": {
        "prompt": REFLEXION_COT_INSTRUCTION_TRIVIAQA,
        "fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "ambignq": {
        "prompt": REFLEXION_COT_INSTRUCTION_AMBIGNQ,
        "fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "gsm8k": {
        "prompt": REFLEXION_COT_INSTRUCTION_GSM8K,
        "fewshot": GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "svamp": {
        "prompt": REFLEXION_COT_INSTRUCTION_SVAMP,
        "fewshot": SVAMP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "tabmwp": {
        "prompt": REFLEXION_COT_INSTRUCTION_TABMWP,
        "fewshot": TABMWP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "humaneval": {
        "prompt": REFLEXION_COT_INSTRUCTION_HUMANEVAL,
        "fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
    "mbpp": {
        "prompt": REFLEXION_COT_INSTRUCTION_MBPP,
        "fewshot": MBPP_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    },
}


class ReflexionCoT(BaseAgent):
    """ReflexionCoT agent that uses Chain of Thought with reflection."""

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_trials: int = 3,
        verbose: bool = False,
        verbosity_level: int = 1,
        **kwargs,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose)
        if benchmark not in COT_BENCHMARK_CONFIG:
            raise ValueError(
                f"Benchmark '{benchmark}' not supported. Available: {list(COT_BENCHMARK_CONFIG.keys())}"
            )

        self.config = COT_BENCHMARK_CONFIG[benchmark]
        self.max_trials = max_trials
        self.verbosity_level = verbosity_level

    def _extract_answer(self, text: str) -> str:
        """Extract answer from Finish[answer] format."""
        import re
        match = re.search(r"Finish\[(.*?)\]", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def _print_llm_io(self, trial_num: int, prompt: str, response: str, response_time: float, tokens: int = 0, cost: float = 0.0, answer: str = ""):
        """Print LLM input/output details and metrics."""
        if not self.verbose or self.verbosity_level < 2:
            return
        
        print(f"\n{'='*50}")
        print(f"TRIAL {trial_num}")
        print(f"{'='*50}")
        if answer:
            print(f"🎯 ANSWER: {answer}")
        print(f"⏱️  Time: {response_time:.2f}s")
        if tokens > 0:
            print(f"🔢 Tokens: {tokens}")
        if cost > 0:
            print(f"💰 Cost: ${cost:.4f}")
        
        print(f"\n📝 LLM INPUT (Trial {trial_num}):")
        print(f"{'─'*30}")
        print(prompt)
        print(f"\n🤖 LLM OUTPUT (Trial {trial_num}):")
        print(f"{'─'*30}")
        print(response[:500] + "..." if len(response) > 500 else response)
        print(f"{'='*50}")

    def _generate_cot_response(self, question: str, trial_num: int = 1) -> Tuple[str, Dict[str, Any]]:
        """Generate a single CoT response."""
        start_time = time.time()
        
        # Build prompt with fewshot examples and question
        prompt = f"{self.config['fewshot']}\n\n{self.config['prompt']}\n\nQuestion: {question}"
        
        # Generate response
        response = self.llm(prompt)
        response_text = response.output_text
        
        # Extract answer
        answer = self._extract_answer(response_text)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens = response.total_tokens
        cost = response.total_cost
        
        metrics = {
            "trial": trial_num,
            "total_time": total_time,
            "total_tokens": tokens,
            "total_cost": cost,
            "response_text": response_text,
            "answer": answer,
        }
        
        # Print verbose output
        self._print_llm_io(trial_num, prompt, response_text, response.prompt_time, tokens, cost, answer)
        
        return answer, metrics

    def generate(self, question: str, **kwargs) -> Dict[str, Any]:
        """Generate answer using ReflexionCoT approach."""
        start_time = time.time()
        total_tokens = total_cost = 0
        trials = []
        best_answer = ""
        
        if self.verbose:
            print(f"\n🚀 Starting ReflexionCoT Agent for benchmark: {self.benchmark}")
            print(f"❓ Question: {question}")
            print(f"📊 Max trials: {self.max_trials}")
            if self.verbosity_level >= 2:
                print(f"🔍 Verbosity level: {self.verbosity_level} (LLM I/O enabled)")
        
        # Generate initial response
        answer, metrics = self._generate_cot_response(question, trial_num=1)
        trials.append(metrics)
        total_tokens += metrics["total_tokens"]
        total_cost += metrics["total_cost"]
        best_answer = answer
        
        # Reflection and improvement trials
        for trial in range(2, self.max_trials + 1):
            if self.verbose:
                print(f"\n🔄 Starting Trial {trial} with reflection...")
            
            # Build reflection prompt
            previous_trials_text = "\n\n".join([
                f"Trial {t['trial']}:\n{t['response_text']}" for t in trials
            ])
            
            reflection_prompt = f"""You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous reasoning trial in which you were given a question to answer.

Previous trials:
{previous_trials_text}

Based on the previous trials, reflect on what went wrong and how to improve. Then provide a new, improved answer.

{self.config['prompt']}

Question: {question}"""
            
            # Generate improved response
            answer, metrics = self._generate_cot_response(question, trial_num=trial)
            trials.append(metrics)
            total_tokens += metrics["total_tokens"]
            total_cost += metrics["total_cost"]
            best_answer = answer
        
        total_time = time.time() - start_time
        
        # Log overall metrics
        if self.verbose:
            print(f"\n📈 FINAL METRICS:")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🔢 Total tokens: {total_tokens}")
            print(f"💰 Total cost: ${total_cost:.4f}")
            print(f"🔄 Trials taken: {len(trials)}")
            print(f"🎯 Final answer: {best_answer}")
        
        return {
            "answer": best_answer,
            "trials": trials,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "trials_taken": len(trials),
            },
        }

    @staticmethod
    def get_fewshots(benchmark: str) -> str:
        return COT_BENCHMARK_CONFIG.get(benchmark, {}).get("fewshot", "")

    @staticmethod
    def get_prompts(benchmark: str) -> str:
        return COT_BENCHMARK_CONFIG.get(benchmark, {}).get("prompt", "")

    @staticmethod
    def list_benchmarks() -> List[str]:
        return list(COT_BENCHMARK_CONFIG.keys()) 