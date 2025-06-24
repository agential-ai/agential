"""Minimal Reflexion Agent - No Logging Version."""

from typing import List, Dict, Any, Tuple, Optional
import re
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from agential.core.llm import BaseLLM
from agential.agents.base import BaseAgent
from agential.utils.general import safe_execute
from agential.agents.reflexion.prompts import (
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_INSTRUCTION_FEVER,
    REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
    REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_INSTRUCTION_MBPP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
    REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
    AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
    FEVER_FEWSHOT_EXAMPLES_REACT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
    SVAMP_FEWSHOT_EXAMPLES_REACT,
    TABMWP_FEWSHOT_EXAMPLES_REACT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
    MBPP_FEWSHOT_EXAMPLES_REACT,
)

# Initialize Rich console
console = Console()


# ============================================================================
# BENCHMARK HANDLERS
# ============================================================================

class BenchmarkHandler:
    """Base class for benchmark-specific handling."""
    
    def evaluate_answer(self, answer: str, key: str) -> bool:
        """Evaluate if answer is correct."""
        raise NotImplementedError
    
    def handle_action(self, action_type: str, query: str) -> Tuple[str, bool]:
        """Handle benchmark-specific actions."""
        raise NotImplementedError


class QAHandler(BenchmarkHandler):
    """Handler for QA benchmarks (HotpotQA, FEVER, TriviaQA, AmbigNQ)."""
    
    def evaluate_answer(self, answer: str, key: str) -> bool:
        return answer.strip() == key.strip()
    
    def handle_action(self, action_type: str, query: str) -> Tuple[str, bool]:
        if action_type.lower() == "finish":
            return query, True
        return "Invalid Action. Valid Actions are Search[entity], Lookup[keyword], and Finish[answer].", False


class MathHandler(BenchmarkHandler):
    """Handler for Math benchmarks (GSM8K, SVAMP, TabMWP)."""
    
    def evaluate_answer(self, answer: str, key: str) -> bool:
        try:
            answer_num = float(answer.strip())
            key_num = float(key.strip())
            return abs(answer_num - key_num) < 1e-6
        except:
            return answer.strip() == key.strip()
    
    def handle_action(self, action_type: str, query: str) -> Tuple[str, bool]:
        if action_type.lower() == "finish":
            return query, True
        elif action_type.lower() == "calculate":
            code_answer, execution_status = safe_execute(query)
            return f"Execution Status: {execution_status}\nOutput: answer = {code_answer[0]}", False
        return "Invalid Action. Valid Actions are Calculate[code] and Finish[answer].", False


class CodeHandler(BenchmarkHandler):
    """Handler for Code benchmarks (HumanEval, MBPP)."""
    
    def evaluate_answer(self, answer: str, key: str) -> bool:
        return answer.strip() == "Done"
    
    def handle_action(self, action_type: str, query: str) -> Tuple[str, bool]:
        if action_type.lower() == "finish":
            return query, True
        elif action_type.lower() == "implement":
            code_answer, execution_status = safe_execute(query)
            return f"Execution Status: {execution_status}\nOutput: {code_answer[0]}", False
        return "Invalid Action. Valid Actions are Implement[code] and Finish[answer].", False


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

BENCHMARK_CONFIG = {
    "hotpotqa": {
        "prompt": REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        "handler": QAHandler(),
    },
    "fever": {
        "prompt": REFLEXION_REACT_INSTRUCTION_FEVER,
        "fewshot": FEVER_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_FEVER,
        "handler": QAHandler(),
    },
    "triviaqa": {
        "prompt": REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
        "fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TRIVIAQA,
        "handler": QAHandler(),
    },
    "ambignq": {
        "prompt": REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
        "fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_AMBIGNQ,
        "handler": QAHandler(),
    },
    "gsm8k": {
        "prompt": REFLEXION_REACT_INSTRUCTION_GSM8K,
        "fewshot": GSM8K_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        "handler": MathHandler(),
    },
    "svamp": {
        "prompt": REFLEXION_REACT_INSTRUCTION_SVAMP,
        "fewshot": SVAMP_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_SVAMP,
        "handler": MathHandler(),
    },
    "tabmwp": {
        "prompt": REFLEXION_REACT_INSTRUCTION_TABMWP,
        "fewshot": TABMWP_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_TABMWP,
        "handler": MathHandler(),
    },
    "humaneval": {
        "prompt": REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
        "fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_HUMANEVAL,
        "handler": CodeHandler(),
    },
    "mbpp": {
        "prompt": REFLEXION_REACT_INSTRUCTION_MBPP,
        "fewshot": MBPP_FEWSHOT_EXAMPLES_REACT,
        "reflect_prompt": REFLEXION_REACT_REFLECT_INSTRUCTION_MBPP,
        "handler": CodeHandler(),
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_halted(finished: bool, step_idx: int, max_steps: int) -> bool:
    """Check if the agent should halt."""
    return finished or step_idx >= max_steps


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class Reflexion(BaseAgent):
    """Simple Reflexion agent that uses configuration-driven approach."""

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        max_trials: int = 3,
        max_reflections: int = 3,
        reflect_strategy: Optional[str] = "last_attempt_and_reflexion",
        truncate_length: Optional[int] = None,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=False)
        if benchmark not in BENCHMARK_CONFIG:
            raise ValueError(
                f"Benchmark '{benchmark}' not supported. Available: {list(BENCHMARK_CONFIG.keys())}"
            )

        self.config = BENCHMARK_CONFIG[benchmark]
        self.max_steps = max_steps
        self.max_trials = max_trials
        self.max_reflections = max_reflections
        self.reflect_strategy = reflect_strategy
        self.handler = self.config["handler"]
        self.truncate_length = truncate_length

    def log_llm_io(self, response, context: str = ""):
        """Log LLM input and output using Rich.
        
        Args:
            response: The LLM response object
            context: Context string for the log
        """
        input_text = str(response.input_text)
        output_text = response.output_text
        
        # Truncate long texts for display if truncate_length is specified
        if self.truncate_length is not None:
            if len(input_text) > self.truncate_length:
                input_text = input_text[:self.truncate_length] + "..."
            if len(output_text) > self.truncate_length:
                output_text = output_text[:self.truncate_length] + "..."
        
        console.print(Panel(
            f"[bold blue]LLM {context}[/bold blue]\n\n"
            f"[bold green]INPUT:[/bold green]\n{input_text}\n\n"
            f"[bold yellow]OUTPUT:[/bold yellow]\n{output_text}",
            title="🤖 LLM Call",
            border_style="blue"
        ))

    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action string into action_type and query."""
        match = re.match(r"^(\w+)\[(.+)\]$", action)
        return (match.group(1), match.group(2)) if match else ("", "")

    def generate_reflection(self, question: str, scratchpad: str) -> str:
        """Generate reflection based on the failed attempt using benchmark-specific prompt."""
        reflection_prompt = self.config["reflect_prompt"].format(
            question=question,
            scratchpad=scratchpad
        )
        
        response = self.llm(reflection_prompt)
        self.log_llm_io(response, "Reflection Generation")
        return response.output_text.strip()

    def generate(self, question: str, key: str = "") -> Dict[str, Any]:
        """Generate answer using reflexion approach with multiple trials."""
        start_time = time.time()
        total_tokens = total_cost = 0
        reflections = ""
        all_trials = []
        
        for trial in range(1, self.max_trials + 1):
            trial_start = time.time()
            trial_tokens = trial_cost = 0
            scratchpad, answer, steps, step_metrics = "", "", [], []
            finished = False

            for idx in range(1, self.max_steps + 1):
                step_start = time.time()

                # Build the full prompt using the template
                full_prompt = self.config["prompt"].format(
                    examples=self.config["fewshot"],
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=self.max_steps
                )

                # Generate response
                response = self.llm(full_prompt)
                self.log_llm_io(response, f"Trial {trial}, Step {idx}")
                
                # Parse the response to extract thought and action
                response_text = response.output_text
                lines = response_text.split('\n')
                
                thought = ""
                action_raw = ""
                
                for line in lines:
                    if line.strip().startswith("Thought"):
                        thought = line.split(":", 1)[1].strip() if ":" in line else line.split(" ", 1)[1].strip()
                    elif line.strip().startswith("Action"):
                        action_raw = line.split(":", 1)[1].strip() if ":" in line else line.split(" ", 1)[1].strip()
                        break
                
                scratchpad += f"\nThought {idx}: {thought}"
                scratchpad += f"\nAction {idx}: {action_raw}"

                # Parse action
                action_type, query = self.parse_action(action_raw)
                
                # Handle observation using benchmark-specific handler
                scratchpad += f"\nObservation {idx}: "
                obs, finished = self.handler.handle_action(action_type, query)
                scratchpad += obs

                if finished:
                    answer = query

                # Calculate step metrics
                step_tokens = response.total_tokens
                step_cost = response.total_cost
                step_time = time.time() - step_start

                # Update totals
                total_tokens += step_tokens
                total_cost += step_cost
                trial_tokens += step_tokens
                trial_cost += step_cost

                step_metrics.append({
                    "step": idx,
                    "total_step_time": step_time,
                    "total_step_tokens": step_tokens,
                    "total_step_cost": step_cost,
                })

                steps.append({
                    "thought": thought,
                    "action_type": action_type,
                    "query": query,
                    "observation": obs,
                    "answer": answer,
                })

                if finished:
                    break

            # Check if answer is correct using benchmark-specific handler
            correct = self.handler.evaluate_answer(answer, key) if key else False
            
            # Check if we should reflect
            should_reflect = (
                self.reflect_strategy is not None 
                and not correct 
                and trial < self.max_trials
                and is_halted(finished, len(steps), self.max_steps)
            )
            
            trial_time = time.time() - trial_start
            
            trial_data = {
                "trial": trial,
                "answer": answer,
                "correct": correct,
                "steps": steps,
                "scratchpad": scratchpad,
                "trial_time": trial_time,
                "trial_tokens": trial_tokens,
                "trial_cost": trial_cost,
                "step_metrics": step_metrics,
            }
            
            all_trials.append(trial_data)
            
            # Generate reflection if needed
            if should_reflect:
                reflection = self.generate_reflection(question, scratchpad)
                reflections += f"\n\nReflection {trial}: {reflection}"
            
            # Stop if correct
            if correct:
                break

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "correct": correct,
            "trials": all_trials,
            "reflections": reflections,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "trials_taken": len(all_trials),
            },
        }

    @staticmethod
    def get_fewshots(benchmark: str) -> str:
        return BENCHMARK_CONFIG.get(benchmark, {}).get("fewshot", "")

    @staticmethod
    def get_prompts(benchmark: str) -> str:
        return BENCHMARK_CONFIG.get(benchmark, {}).get("prompt", "")

    @staticmethod
    def list_benchmarks() -> List[str]:
        return list(BENCHMARK_CONFIG.keys())

    @staticmethod
    def add_benchmark(
        name: str,
        prompt: str,
        fewshot: str,
        reflect_prompt: str,
        handler: BenchmarkHandler,
    ):
        """Add a new benchmark configuration."""
        BENCHMARK_CONFIG[name] = {
            "prompt": prompt,
            "fewshot": fewshot,
            "reflect_prompt": reflect_prompt,
            "handler": handler,
        } 