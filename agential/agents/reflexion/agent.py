"""Minimal Reflexion Agent."""

from typing import List, Dict, Any, Tuple, Callable, Optional
import re
from agential.agents.base.agent import BaseAgent
from agential.core.llm import BaseLLM
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


def handle_qa_action(action_type: str, query: str) -> Tuple[str, bool]:
    """Handle QA benchmark actions."""
    if action_type.lower() == "finish":
        return query, True
    else:
        return (
            "Invalid Action. Valid Actions are Search[entity], Lookup[keyword], and Finish[answer].",
            False,
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
    else:
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
    else:
        return (
            "Invalid Action. Valid Actions are Implement[code] and Finish[answer].",
            False,
        )


# Action handlers mapping
ACTION_HANDLERS = {
    "qa": handle_qa_action,
    "math": handle_math_action,
    "code": handle_code_action,
}


class ReflexionAgent(BaseAgent):
    """Simple Reflexion agent that uses configuration-driven approach."""

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        debug_mode: bool = False,
        **kwargs,
    ):
        super().__init__(llm=llm, benchmark=benchmark)
        if benchmark not in BENCHMARK_CONFIG:
            raise ValueError(
                f"Benchmark '{benchmark}' not supported. Available: {list(BENCHMARK_CONFIG.keys())}"
            )

        self.config = BENCHMARK_CONFIG[benchmark]
        self.max_steps = max_steps
        self.debug_mode = debug_mode
        self.action_handler = ACTION_HANDLERS[self.config["action_handler"]]

    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action string into action_type and query."""
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, action)
        return (match.group(1), match.group(2)) if match else ("", "")

    def generate(self, question: str, **kwargs) -> Dict[str, Any]:
        """Generate answer using reflexion approach."""
        scratchpad = question
        answer = ""
        steps = []

        for idx in range(1, self.max_steps + 1):
            # Thought
            scratchpad += f"\nThought {idx}: "
            thought = self.llm(scratchpad).output_text.split("Action")[0].strip()
            scratchpad += thought

            # Action
            scratchpad += f"\nAction {idx}: "
            action_raw = self.llm(scratchpad).output_text.split("Observation")[0]
            action_type, query = self.parse_action(action_raw)
            scratchpad += f"{action_type}[{query}]"

            # Observation
            scratchpad += f"\nObservation {idx}: "
            obs, finished = self.action_handler(action_type, query)
            scratchpad += obs

            if finished:
                answer = query

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
                break

        return {"answer": answer, "steps": steps}

    @staticmethod
    def get_fewshots(benchmark: str) -> str:
        """Get fewshot examples for a benchmark."""
        return BENCHMARK_CONFIG.get(benchmark, {}).get("fewshot", "")

    @staticmethod
    def get_prompts(benchmark: str) -> str:
        """Get prompt for a benchmark."""
        return BENCHMARK_CONFIG.get(benchmark, {}).get("prompt", "")

    @staticmethod
    def list_benchmarks() -> List[str]:
        """List all supported benchmarks."""
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
        """Add a new benchmark configuration.

        Args:
            name: Benchmark name
            prompt: Instruction prompt
            fewshot: Fewshot examples
            actions: List of valid actions
            action_handler: Handler type ("qa", "math", "code") or custom function
            handler_func: Custom handler function (optional)
        """
        BENCHMARK_CONFIG[name] = {
            "prompt": prompt,
            "fewshot": fewshot,
            "actions": actions,
            "action_handler": action_handler,
        }

        if handler_func is not None and action_handler not in ACTION_HANDLERS:
            ACTION_HANDLERS[action_handler] = handler_func
