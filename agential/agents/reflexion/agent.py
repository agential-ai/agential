"""Minimal Reflexion Agent."""

from typing import List, Dict, Any, Tuple, Callable, Optional
import re
import time
import logging
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
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
)

# Configuration-driven approach - everything in one place
BENCHMARK_CONFIG = {
    "hotpotqa": {
        "prompt": REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        "fewshot": HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Search", "Lookup", "Finish"],
        "action_handler": "qa",
    },
    "fever": {
        "prompt": REFLEXION_REACT_INSTRUCTION_FEVER,
        "fewshot": FEVER_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Search", "Lookup", "Finish"],
        "action_handler": "qa",
    },
    "triviaqa": {
        "prompt": REFLEXION_REACT_INSTRUCTION_TRIVIAQA,
        "fewshot": TRIVIAQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Search", "Lookup", "Finish"],
        "action_handler": "qa",
    },
    "ambignq": {
        "prompt": REFLEXION_REACT_INSTRUCTION_AMBIGNQ,
        "fewshot": AMBIGNQ_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Search", "Lookup", "Finish"],
        "action_handler": "qa",
    },
    "gsm8k": {
        "prompt": REFLEXION_REACT_INSTRUCTION_GSM8K,
        "fewshot": GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Calculate", "Finish"],
        "action_handler": "math",
    },
    "svamp": {
        "prompt": REFLEXION_REACT_INSTRUCTION_SVAMP,
        "fewshot": SVAMP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Calculate", "Finish"],
        "action_handler": "math",
    },
    "tabmwp": {
        "prompt": REFLEXION_REACT_INSTRUCTION_TABMWP,
        "fewshot": TABMWP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Calculate", "Finish"],
        "action_handler": "math",
    },
    "humaneval": {
        "prompt": REFLEXION_REACT_INSTRUCTION_HUMANEVAL,
        "fewshot": HUMANEVAL_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Implement", "Finish"],
        "action_handler": "code",
    },
    "mbpp": {
        "prompt": REFLEXION_REACT_INSTRUCTION_MBPP,
        "fewshot": MBPP_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        "actions": ["Implement", "Finish"],
        "action_handler": "code",
    },
}


# Simplified action handlers
def handle_qa_action(action_type: str, query: str) -> Tuple[str, bool]:
    """Handle QA benchmark actions."""
    return (
        (query, True)
        if action_type.lower() == "finish"
        else (
            "Invalid Action. Valid Actions are Search[entity], Lookup[keyword], and Finish[answer].",
            False,
        )
    )


def handle_math_action(action_type: str, query: str) -> Tuple[str, bool]:
    """Handle Math benchmark actions."""
    if action_type.lower() == "finish":
        return query, True
    elif action_type.lower() == "calculate":
        code_answer, execution_status = safe_execute(query)
        return (
            f"Execution Status: {execution_status}\nOutput: answer = {code_answer[0]}",
            False,
        )
    return (
        "Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
        False,
    )


def handle_code_action(action_type: str, query: str) -> Tuple[str, bool]:
    """Handle Code benchmark actions."""
    if action_type.lower() == "finish":
        return query, True
    elif action_type.lower() == "implement":
        code_answer, execution_status = safe_execute(query)
        return f"Execution Status: {execution_status}\nOutput: {code_answer[0]}", False
    return (
        "Invalid Action. Valid Actions are Implement[code] and Finish[answer].",
        False,
    )


ACTION_HANDLERS = {
    "qa": handle_qa_action,
    "math": handle_math_action,
    "code": handle_code_action,
}


class Reflexion(BaseAgent):
    """Simple Reflexion agent that uses configuration-driven approach."""

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose)
        if benchmark not in BENCHMARK_CONFIG:
            raise ValueError(
                f"Benchmark '{benchmark}' not supported. Available: {list(BENCHMARK_CONFIG.keys())}"
            )

        self.config = BENCHMARK_CONFIG[benchmark]
        self.max_steps = max_steps
        self.action_handler = ACTION_HANDLERS[self.config["action_handler"]]

    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action string into action_type and query."""
        match = re.match(r"^(\w+)\[(.+)\]$", action)
        return (match.group(1), match.group(2)) if match else ("", "")

    def _print_verbose(self, step: int, thought: str, action_type: str, query: str, obs: str):
        """Print verbose output for the current step."""
        if not self.verbose:
            return
        
        print(f"\n{'='*50}")
        print(f"STEP {step}")
        print(f"{'='*50}")
        print(f"🤔 THOUGHT: {thought}")
        print(f"⚡ ACTION: {action_type}[{query}]")
        print(f"👁️  OBSERVATION: {obs}")
        print(f"{'='*50}")

    def _print_llm_io(self, step: int, prompt: str, response: str, response_time: float):
        """Print LLM input/output details."""
        if not self.verbose:
            return
        
        print(f"\n📝 LLM INPUT (Step {step}):")
        print(f"{'─'*30}")
        print(prompt)
        print(f"\n🤖 LLM OUTPUT (Step {step}):")
        print(f"{'─'*30}")
        print(response)
        print(f"⏱️  Response time: {response_time:.2f}s")

    def _log_step(self, step_metrics):
        """Log step metrics if verbose mode is enabled."""
        if not self.verbose:
            return
        for metric in step_metrics:
            logging.info(
                f"Step {metric['step']}: "
                f"Time={metric['total_step_time']:.2f}s, "
                f"Tokens={metric['total_step_tokens']}, "
                f"Cost=${metric['total_step_cost']:.4f}"
            )

    def generate(self, question: str, **kwargs) -> Dict[str, Any]:
        """Generate answer using reflexion approach."""
        start_time = time.time()
        total_tokens = total_cost = 0
        scratchpad, answer, steps, step_metrics = question, "", [], []

        if self.verbose:
            print(f"\n🚀 Starting Reflexion Agent for benchmark: {self.benchmark}")
            print(f"❓ Question: {question}")
            print(f"📊 Max steps: {self.max_steps}")

        for idx in range(1, self.max_steps + 1):
            step_start = time.time()

            # Generate thought
            scratchpad += f"\nThought {idx}: "
            thought_response = self.llm(scratchpad)
            thought = thought_response.output_text.split("Action")[0].strip()
            scratchpad += thought

            # Generate action
            scratchpad += f"\nAction {idx}: "
            action_response = self.llm(scratchpad)
            action_raw = action_response.output_text.split("Observation")[0]
            action_type, query = self.parse_action(action_raw)
            scratchpad += f"{action_type}[{query}]"

            # Handle observation
            scratchpad += f"\nObservation {idx}: "
            obs, finished = self.action_handler(action_type, query)
            scratchpad += obs

            if finished:
                answer = query

            # Print verbose output
            self._print_verbose(idx, thought, action_type, query, obs)
            self._print_llm_io(idx, scratchpad, action_response.output_text, action_response.prompt_time)

            # Calculate step metrics using LLM response data
            step_tokens = thought_response.total_tokens + action_response.total_tokens
            step_cost = thought_response.total_cost + action_response.total_cost
            step_time = time.time() - step_start

            # Update totals
            total_tokens += step_tokens
            total_cost += step_cost

            step_metrics.append(
                {
                    "step": idx,
                    "total_step_time": step_time,
                    "thought_tokens": thought_response.total_tokens,
                    "action_tokens": action_response.total_tokens,
                    "total_step_tokens": step_tokens,
                    "thought_cost": thought_response.total_cost,
                    "action_cost": action_response.total_cost,
                    "total_step_cost": step_cost,
                }
            )

            steps.append(
                {
                    "thought": thought,
                    "action_type": action_type,
                    "query": query,
                    "observation": obs,
                    "answer": answer,
                }
            )

            if finished:
                if self.verbose:
                    print(f"\n✅ Finished in {idx} steps!")
                break

        total_time = time.time() - start_time

        # Log metrics
        if self.verbose:
            print(f"\n📈 FINAL METRICS:")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🔢 Total tokens: {total_tokens}")
            print(f"💰 Total cost: ${total_cost:.4f}")
            print(f"👣 Steps taken: {len(steps)}")
            print(f"🎯 Final answer: {answer}")
            self._log_step(step_metrics)

        return {
            "answer": answer,
            "steps": steps,
            "metrics": {
                "total_time": total_time,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "steps_taken": len(steps),
                "step_metrics": step_metrics,
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
        actions: List[str],
        action_handler: str,
        handler_func: Optional[Callable] = None,
    ):
        """Add a new benchmark configuration."""
        BENCHMARK_CONFIG[name] = {
            "prompt": prompt,
            "fewshot": fewshot,
            "actions": actions,
            "action_handler": action_handler,
        }
        if handler_func is not None and action_handler not in ACTION_HANDLERS:
            ACTION_HANDLERS[action_handler] = handler_func
