"""Minimal Reflexion Agent."""

from typing import List, Dict, Any, Tuple, Callable, Optional
import re
import time
from rich.console import Console
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
        verbose_level: int = 1,
    ):
        super().__init__(llm=llm, benchmark=benchmark, verbose=verbose)
        if benchmark not in BENCHMARK_CONFIG:
            raise ValueError(
                f"Benchmark '{benchmark}' not supported. Available: {list(BENCHMARK_CONFIG.keys())}"
            )

        self.config = BENCHMARK_CONFIG[benchmark]
        self.max_steps = max_steps
        self.verbose_level = verbose_level
        self.action_handler = ACTION_HANDLERS[self.config["action_handler"]]
        self.console = Console() if verbose else None

    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action string into action_type and query."""
        match = re.match(r"^(\w+)\[(.+)\]$", action)
        return (match.group(1), match.group(2)) if match else ("", "")

    def _print_step(
        self,
        step: int,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        thought_response,
        action_response,
    ):
        """Print all verbose output for a step."""
        if not self.verbose or not self.console:
            return

        # Print step header
        self.console.print(f"\n[bold blue]Step {step}[/bold blue]")
        self.console.print("─" * 50)

        # Print thought
        self.console.print(f"[bold green]💭 Thought:[/bold green]")
        self.console.print(f"   {thought}")

        # Print action
        self.console.print(
            f"[bold yellow]🔧 Action:[/bold yellow] {action_type}[{query}]"
        )

        # Print observation
        self.console.print(f"[bold magenta]👁️  Observation:[/bold magenta]")
        self.console.print(f"   {obs}")

        # Show LLM I/O if verbosity level >= 2
        if self.verbose_level >= 2:
            if thought_response:
                self.console.print(f"\n[bold cyan]📝 LLM INPUT (THOUGHT):[/bold cyan]")
                self.console.print("─" * 30)
                self.console.print(str(thought_response.input_text))
                self.console.print(f"\n[bold cyan]🤖 LLM OUTPUT (THOUGHT):[/bold cyan]")
                self.console.print("─" * 30)
                self.console.print(thought_response.output_text)
                self.console.print(
                    f"[dim]⏱️  Time: {thought_response.prompt_time:.2f}s | 🔢 Tokens: {thought_response.total_tokens} | 💰 Cost: ${thought_response.total_cost:.4f}[/dim]"
                )

            if action_response:
                self.console.print(f"\n[bold cyan]📝 LLM INPUT (ACTION):[/bold cyan]")
                self.console.print("─" * 30)
                self.console.print(str(action_response.input_text))
                self.console.print(f"\n[bold cyan]🤖 LLM OUTPUT (ACTION):[/bold cyan]")
                self.console.print("─" * 30)
                self.console.print(action_response.output_text)
                self.console.print(
                    f"[dim]⏱️  Time: {action_response.prompt_time:.2f}s | 🔢 Tokens: {action_response.total_tokens} | 💰 Cost: ${action_response.total_cost:.4f}[/dim]"
                )

        self.console.print("─" * 50)

    def generate(self, question: str) -> Dict[str, Any]:
        """Generate answer using reflexion approach."""
        start_time = time.time()
        total_tokens = total_cost = 0
        scratchpad, answer, steps, step_metrics = question, "", [], []

        if self.verbose and self.console:
            self.console.print(
                f"\n[bold blue]🚀 Starting Reflexion Agent for benchmark: {self.benchmark}[/bold blue]"
            )
            self.console.print(f"[bold yellow]❓ Question:[/bold yellow] {question}")
            self.console.print(
                f"[bold yellow]📊 Max steps:[/bold yellow] {self.max_steps}"
            )
            if self.verbose_level >= 2:
                self.console.print(
                    f"[bold yellow]🔍 Verbosity level:[/bold yellow] {self.verbose_level} (LLM I/O enabled)"
                )

        for idx in range(1, self.max_steps + 1):
            step_start = time.time()

            # Generate thought
            thought_prompt = scratchpad + f"\nThought {idx}: "
            thought_response = self.llm(thought_prompt)
            thought = thought_response.output_text.split("Action")[0].strip()
            scratchpad += thought

            # Generate action
            action_prompt = scratchpad + f"\nAction {idx}: "
            action_response = self.llm(action_prompt)
            action_raw = action_response.output_text.split("Observation")[0]
            action_type, query = self.parse_action(action_raw)
            scratchpad += f"{action_type}[{query}]"

            # Handle observation
            scratchpad += f"\nObservation {idx}: "
            obs, finished = self.action_handler(action_type, query)
            scratchpad += obs

            if finished:
                answer = query

            # Print all verbose output for this step
            self._print_step(
                idx, thought, action_type, query, obs, thought_response, action_response
            )

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
                if self.verbose and self.console:
                    self.console.print(
                        f"\n[bold green]✅ Finished in {idx} steps![/bold green]"
                    )
                break

        total_time = time.time() - start_time

        # Log metrics
        if self.verbose and self.console:
            self.console.print(f"\n[bold blue]📈 FINAL METRICS:[/bold blue]")
            self.console.print(f"[dim]⏱️  Total time: {total_time:.2f}s[/dim]")
            self.console.print(f"[dim]🔢 Total tokens: {total_tokens}[/dim]")
            self.console.print(f"[dim]💰 Total cost: ${total_cost:.4f}[/dim]")
            self.console.print(f"[dim]👣 Steps taken: {len(steps)}[/dim]")
            self.console.print(f"[bold green]🎯 Final answer: {answer}[/bold green]")

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
