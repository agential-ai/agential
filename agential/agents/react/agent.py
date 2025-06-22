"""Scalable ReAct Agent with Plugin-Based Architecture.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct

This version makes it extremely easy to add new benchmarks without modifying the core agent.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from agential.agents.base.agent import BaseAgent
from agential.agents.react.prompts import (
    REACT_INSTRUCTION_AMBIGNQ,
    REACT_INSTRUCTION_FEVER,
    REACT_INSTRUCTION_GSM8K,
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_HUMANEVAL,
    REACT_INSTRUCTION_MBPP,
    REACT_INSTRUCTION_SVAMP,
    REACT_INSTRUCTION_TABMWP,
    REACT_INSTRUCTION_TRIVIAQA,
)
from agential.agents.react.handlers import (
    BENCHMARK_HANDLERS,
    QAHandler,
    MathHandler,
    CodeHandler,
)
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.llm import BaseLLM, Response
from agential.utils.parse import remove_newline

from rich.console import Console


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# ReAct-specific constants
REACT_FEWSHOT_TYPE = FewShotType.REACT

# Available benchmarks for ReAct
REACT_BENCHMARKS = [
    Benchmarks.HOTPOTQA,
    Benchmarks.FEVER,
    Benchmarks.TRIVIAQA,
    Benchmarks.AMBIGNQ,
    Benchmarks.GSM8K,
    Benchmarks.SVAMP,
    Benchmarks.TABMWP,
    Benchmarks.HUMANEVAL,
    Benchmarks.MBPP,
]

# Benchmark configurations - makes it easy to add new benchmarks
BENCHMARK_CONFIGS = {
    # QA Benchmarks
    Benchmarks.HOTPOTQA: {"prompt": REACT_INSTRUCTION_HOTPOTQA},
    Benchmarks.FEVER: {"prompt": REACT_INSTRUCTION_FEVER},
    Benchmarks.TRIVIAQA: {"prompt": REACT_INSTRUCTION_TRIVIAQA},
    Benchmarks.AMBIGNQ: {"prompt": REACT_INSTRUCTION_AMBIGNQ},
    
    # Math Benchmarks
    Benchmarks.GSM8K: {"prompt": REACT_INSTRUCTION_GSM8K},
    Benchmarks.SVAMP: {"prompt": REACT_INSTRUCTION_SVAMP},
    Benchmarks.TABMWP: {"prompt": REACT_INSTRUCTION_TABMWP},
    
    # Code Benchmarks
    Benchmarks.HUMANEVAL: {"prompt": REACT_INSTRUCTION_HUMANEVAL},
    Benchmarks.MBPP: {"prompt": REACT_INSTRUCTION_MBPP},
}

# Simple prompt mapping for get_prompts method
BENCHMARK_PROMPTS: Dict[str, str] = {
    Benchmarks.HOTPOTQA: REACT_INSTRUCTION_HOTPOTQA,
    Benchmarks.FEVER: REACT_INSTRUCTION_FEVER,
    Benchmarks.TRIVIAQA: REACT_INSTRUCTION_TRIVIAQA,
    Benchmarks.AMBIGNQ: REACT_INSTRUCTION_AMBIGNQ,
    Benchmarks.GSM8K: REACT_INSTRUCTION_GSM8K,
    Benchmarks.SVAMP: REACT_INSTRUCTION_SVAMP,
    Benchmarks.TABMWP: REACT_INSTRUCTION_TABMWP,
    Benchmarks.HUMANEVAL: REACT_INSTRUCTION_HUMANEVAL,
    Benchmarks.MBPP: REACT_INSTRUCTION_MBPP,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_react_fewshots(benchmark: str) -> str:
    """Get ReAct few-shot examples for a benchmark."""
    if benchmark not in BENCHMARK_FEWSHOTS:
        raise ValueError(f"Benchmark '{benchmark}' not found.")
    
    if REACT_FEWSHOT_TYPE not in BENCHMARK_FEWSHOTS[benchmark]:
        raise ValueError(f"ReAct few-shot type not supported for '{benchmark}'.")
    
    return BENCHMARK_FEWSHOTS[benchmark][REACT_FEWSHOT_TYPE]


def add_benchmark(benchmark_name: str, prompt: str, handler_class=None):
    """Add a new benchmark easily.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "my_benchmark")
        prompt: The prompt template for this benchmark
        handler_class: Optional custom handler class. If None, automatically determined.
                      Must inherit from QAHandler, MathHandler, or CodeHandler.
    """
    # Add to configurations
    BENCHMARK_CONFIGS[benchmark_name] = {"prompt": prompt}
    
    # Add to prompts mapping
    BENCHMARK_PROMPTS[benchmark_name] = prompt
    
    # Add to benchmarks list
    if benchmark_name not in REACT_BENCHMARKS:
        REACT_BENCHMARKS.append(benchmark_name)
    
    # Add to handlers registry
    if handler_class is not None:
        # User provided a custom handler
        if not (issubclass(handler_class, QAHandler) or 
                issubclass(handler_class, MathHandler) or 
                issubclass(handler_class, CodeHandler)):
            raise ValueError(f"Handler class must inherit from QAHandler, MathHandler, or CodeHandler")
        BENCHMARK_HANDLERS[benchmark_name] = handler_class
    else:
        # Auto-determine handler type based on benchmark name pattern
        if benchmark_name in [Benchmarks.HOTPOTQA, Benchmarks.FEVER, Benchmarks.TRIVIAQA, Benchmarks.AMBIGNQ]:
            BENCHMARK_HANDLERS[benchmark_name] = QAHandler
        elif benchmark_name in [Benchmarks.GSM8K, Benchmarks.SVAMP, Benchmarks.TABMWP]:
            BENCHMARK_HANDLERS[benchmark_name] = MathHandler
        elif benchmark_name in [Benchmarks.HUMANEVAL, Benchmarks.MBPP]:
            BENCHMARK_HANDLERS[benchmark_name] = CodeHandler
        else:
            # For custom benchmarks, default to QAHandler
            BENCHMARK_HANDLERS[benchmark_name] = QAHandler


# =============================================================================
# OUTPUT CLASSES
# =============================================================================

@dataclass
class ReActStepOutput:
    """ReAct step output - contains one step of reasoning and action."""
    
    thought: str
    action_type: str
    query: str
    observation: str
    answer: str
    external_tool_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.external_tool_info is None:
            self.external_tool_info = {}


@dataclass
class ReActOutput:
    """ReAct output - contains the final answer and all steps."""
    
    answer: str
    total_tokens: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    steps: List[ReActStepOutput] = field(default_factory=list)
    
    @property
    def num_steps(self) -> int:
        """Number of steps taken."""
        return len(self.steps)
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "answer": self.answer,
            "num_steps": self.num_steps,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
        }


# =============================================================================
# LOGGING
# =============================================================================

class AgentLogger:
    """Simple logging for the ReAct agent."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.console = Console() if verbose else None
        self.metrics = {
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
        }
    
    def log_step(self, step_number: int, thought: str, action_type: str, query: str, 
                 thought_response: Response, action_response: Response):
        """Log a single step with metrics."""
        # Update metrics
        step_tokens = (thought_response.prompt_tokens + thought_response.completion_tokens + 
                      action_response.prompt_tokens + action_response.completion_tokens)
        step_cost = (thought_response.prompt_cost + thought_response.completion_cost + 
                    action_response.prompt_cost + action_response.completion_cost)
        
        self.metrics["total_steps"] += 1
        self.metrics["total_tokens"] += step_tokens
        self.metrics["total_cost"] += step_cost
        
        if self.verbose and self.console:
            self.console.print(f"Step {step_number}: {action_type}[{query[:50]}{'...' if len(query) > 50 else ''}]")
    
    def log_finish(self, answer: str):
        """Log the completion of agent execution."""
        if self.verbose and self.console:
            self.console.print(f"🎯 Answer: {answer}")
            self.console.print(f"📊 Steps: {self.metrics['total_steps']}, Tokens: {self.metrics['total_tokens']}, Cost: ${self.metrics['total_cost']:.4f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        return self.metrics


# =============================================================================
# MAIN REACT AGENT
# =============================================================================

class ReAct(BaseAgent):
    """ReAct agent that uses plugin-based handlers for benchmarks.
    
    This architecture makes it extremely easy to add new benchmarks by creating
    handler classes that inherit from QAHandler, MathHandler, or CodeHandler.

    Attributes:
        llm (BaseLLM): Language model for generation
        benchmark (str): The benchmark name
        testing (bool): Whether in testing mode
        handler (BenchmarkHandler): The benchmark-specific handler
        logger (AgentLogger): Logger for colorful output and metrics
    """

    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        testing: bool = False,
        max_steps: int = 6,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the scalable ReAct agent."""
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)
        
        if benchmark not in BENCHMARK_CONFIGS:
            available = list(BENCHMARK_CONFIGS.keys())
            raise ValueError(f"Unsupported benchmark: {benchmark}. Available: {available}")
            
        if benchmark not in BENCHMARK_HANDLERS:
            raise ValueError(f"Handler not found for benchmark: {benchmark}")
            
        handler_class = BENCHMARK_HANDLERS[benchmark]
        self.handler = handler_class(llm=llm, max_steps=max_steps, testing=testing)
        self.max_steps = max_steps
        self.logger = AgentLogger(verbose=verbose)
    
    def _build_prompt(
        self,
        question: str,
        scratchpad: str,
        examples: str,
        additional_keys: Dict[str, str] = {},
    ) -> str:
        """Build the prompt for the agent."""
        return self.handler.get_prompt().format(
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            max_steps=self.max_steps,
            **additional_keys,
        )
    
    def _prompt_agent(
        self,
        question: str,
        scratchpad: str,
        examples: str,
        additional_keys: Dict[str, str] = {},
    ) -> Response:
        """Generate a response from the LLM."""
        prompt = self._build_prompt(question, scratchpad, examples, additional_keys)
        return self.llm(prompt)
    
    def _generate_thought(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, Response]:
        """Generate a thought step."""
        scratchpad += f"\nThought {idx}: "
        out = self._prompt_agent(question, scratchpad, examples, additional_keys)
        thought = remove_newline(out.output_text).split("Action")[0].strip()
        scratchpad += thought
        return scratchpad, thought, out
    
    def _generate_action(
        self,
        idx: int,
        scratchpad: str,
        question: str,
        examples: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str, str, Response]:
        """Generate an action step."""
        scratchpad += f"\nAction {idx}: "
        out = self._prompt_agent(question, scratchpad, examples, additional_keys)
        action = remove_newline(out.output_text).split("Observation")[0]
        action_type, query = self.handler.parse_action(action)
        scratchpad += f"{action_type}[{query}]"
        return scratchpad, action_type, query, out
    
    def _generate_observation(
        self, 
        idx: int, 
        scratchpad: str, 
        action_type: str, 
        query: str
    ) -> Tuple[str, str, str, bool, Dict[str, Any]]:
        """Generate an observation step."""
        scratchpad += f"\nObservation {idx}: "
        obs, answer, finished, external_tool_info = self.handler.handle_observation(
            action_type, query, scratchpad
        )
        scratchpad += obs
        return scratchpad, answer, obs, finished, external_tool_info

    def generate(
        self,
        question: str,
        examples: str = "",
        prompt: str = "",
        additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        reset: bool = True,
    ) -> ReActOutput:
        """Generate a ReAct output by iteratively thinking, acting, and observing."""
        if reset:
            self.reset()
            
        scratchpad = ""
        answer = ""
        finished = False
        idx = 1
        steps = []
        
        while not finished and idx <= self.max_steps:
            # Think
            scratchpad, thought, thought_response = self._generate_thought(
                idx, scratchpad, question, examples, additional_keys
            )
            
            # Act
            scratchpad, action_type, query, action_response = self._generate_action(
                idx, scratchpad, question, examples, additional_keys
            )
            
            # Log the step
            self.logger.log_step(
                idx, thought, action_type, query, 
                thought_response, action_response
            )
            
            # Observe
            scratchpad, answer, obs, finished, external_tool_info = self._generate_observation(
                idx, scratchpad, action_type, query
            )
            
            steps.append(ReActStepOutput(
                thought=thought,
                action_type=action_type,
                query=query,
                observation=obs,
                answer=answer,
                external_tool_info=external_tool_info,
            ))
            
            idx += 1
        
        # Log completion
        self.logger.log_finish(answer)
        
        # Get metrics from logger
        metrics = self.logger.get_metrics()
        
        return ReActOutput(
            answer=answer,
            total_tokens=metrics["total_tokens"],
            total_cost=metrics["total_cost"],
            total_time=metrics["total_time"],
            steps=steps,
        )
    
    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.handler.reset()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics from the logger."""
        return self.logger.get_metrics()
    
    @staticmethod
    def get_fewshots(benchmark: str, fewshot_type: str = "react", **kwargs: Any) -> Dict[str, str]:
        """Get few-shot examples for the benchmark."""
        if benchmark not in BENCHMARK_CONFIGS:
            raise ValueError(f"Benchmark '{benchmark}' not found for ReAct.")
        
        if fewshot_type != "react":
            raise ValueError(f"ReAct only supports 'react' few-shot type, got '{fewshot_type}'.")
        
        benchmark_fewshots = get_react_fewshots(benchmark)
        return {"examples": benchmark_fewshots}
    
    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        """Get prompts for the benchmark."""
        if benchmark not in BENCHMARK_PROMPTS:
            raise ValueError(f"Benchmark '{benchmark}' prompt not found for ReAct.")
        
        return {"prompt": BENCHMARK_PROMPTS[benchmark]}
    
    @staticmethod
    def list_benchmarks() -> List[str]:
        """List all registered benchmarks."""
        return list(BENCHMARK_CONFIGS.keys())
    
    def get_strategy(self, benchmark: str, **kwargs: Any):
        """Required by BaseAgent - not used in ReAct."""
        raise NotImplementedError("ReAct agent does not use strategies.") 