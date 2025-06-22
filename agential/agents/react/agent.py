"""Scalable ReAct Agent with Plugin-Based Architecture.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct

This version makes it extremely easy to add new benchmarks without modifying the core agent.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type
from datetime import datetime

from agential.agents.base.agent import BaseAgent
from agential.agents.react.output import ReActOutput, ReActStepOutput
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
from agential.constants import BENCHMARK_FEWSHOTS, Benchmarks, FewShotType
from agential.core.llm import BaseLLM, Response
from agential.utils.docstore import DocstoreExplorer
from agential.utils.general import safe_execute
from agential.utils.parse import remove_newline
from langchain_community.docstore.wikipedia import Wikipedia

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class AgentLogger:
    """Professional logging for the ReAct agent with colorful output."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.console = Console()
        self.metrics = {
            "start_time": datetime.now(),
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
            "steps": [],
            "benchmark": "",
            "question": ""
        }
    
    def log_start(self, benchmark: str, question: str):
        """Log the start of agent execution."""
        self.metrics["benchmark"] = benchmark
        self.metrics["question"] = question
        
        if not self.verbose:
            return
            
        title = Text("🤖 ReAct Agent Starting", style="bold blue")
        content = f"""
        [bold]Benchmark:[/bold] {benchmark}
        [bold]Question:[/bold] {question[:100]}{'...' if len(question) > 100 else ''}
        [bold]Time:[/bold] {self.metrics['start_time'].strftime('%H:%M:%S')}
        """
        panel = Panel(content, title=title, border_style="blue")
        self.console.print(panel)
    
    def log_step(self, step_number: int, thought: str, action_type: str, query: str, 
                 thought_response: Response, action_response: Response, finished: bool = False):
        """Log a single step with metrics."""
        # Create step metrics
        step_metrics = {
            "step_number": step_number,
            "thought_tokens": thought_response.prompt_tokens + thought_response.completion_tokens,
            "thought_cost": thought_response.prompt_cost + thought_response.completion_cost,
            "thought_time": thought_response.prompt_time,
            "action_tokens": action_response.prompt_tokens + action_response.completion_tokens,
            "action_cost": action_response.prompt_cost + action_response.completion_cost,
            "action_time": action_response.prompt_time,
            "action_type": action_type,
            "finished": finished
        }
        
        # Calculate totals
        step_metrics["total_tokens"] = step_metrics["thought_tokens"] + step_metrics["action_tokens"]
        step_metrics["total_cost"] = step_metrics["thought_cost"] + step_metrics["action_cost"]
        step_metrics["total_time"] = step_metrics["thought_time"] + step_metrics["action_time"]
        
        self.metrics["steps"].append(step_metrics)
        self.metrics["total_steps"] = len(self.metrics["steps"])
        self.metrics["total_tokens"] += step_metrics["total_tokens"]
        self.metrics["total_cost"] += step_metrics["total_cost"]
        self.metrics["total_time"] += step_metrics["total_time"]
        
        if not self.verbose:
            return
            
        # Create step panel
        step_title = f"Step {step_number}"
        if finished:
            step_title += " 🎯 (FINISHED)"
        
        thought_text = Text(f"💭 {thought[:200]}{'...' if len(thought) > 200 else ''}", style="cyan")
        action_text = Text(f"⚡ {action_type}[{query[:100]}{'...' if len(query) > 100 else ''}]", style="yellow")
        
        # Metrics table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold")
        table.add_row("Thought Tokens", f"{step_metrics['thought_tokens']}")
        table.add_row("Action Tokens", f"{step_metrics['action_tokens']}")
        table.add_row("Total Cost", f"${step_metrics['total_cost']:.4f}")
        table.add_row("Step Time", f"{step_metrics['total_time']:.2f}s")
        
        content = f"{thought_text}\n{action_text}\n\n{table}"
        panel = Panel(content, title=step_title, border_style="green" if finished else "blue")
        self.console.print(panel)
    
    def log_observation(self, observation: str, finished: bool = False):
        """Log the observation for the current step."""
        if not self.verbose:
            return
            
        if finished:
            obs_text = Text(f"🎉 {observation}", style="bold green")
        else:
            obs_text = Text(f"👁️ {observation[:200]}{'...' if len(observation) > 200 else ''}", style="magenta")
        self.console.print(obs_text)
    
    def log_finish(self, answer: str):
        """Log the completion of agent execution."""
        self.metrics["end_time"] = datetime.now()
        self.metrics["total_time"] = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        
        if not self.verbose:
            return
            
        # Create summary table
        table = Table(title="📊 Agent Execution Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("Total Steps", str(self.metrics["total_steps"]))
        table.add_row("Total Tokens", f"{self.metrics['total_tokens']:,}")
        table.add_row("Total Cost", f"${self.metrics['total_cost']:.4f}")
        table.add_row("Total Time", f"{self.metrics['total_time']:.2f}s")
        table.add_row("Average Time/Step", f"{self.metrics['total_time']/max(1, self.metrics['total_steps']):.2f}s")
        
        # Final answer
        answer_text = Text(f"🎯 Final Answer: {answer}", style="bold green")
        
        self.console.print(table)
        self.console.print(answer_text)
        
        # Step breakdown
        if self.metrics["steps"]:
            step_table = Table(title="📈 Step Breakdown", show_header=True, header_style="bold blue")
            step_table.add_column("Step", style="cyan")
            step_table.add_column("Action", style="yellow")
            step_table.add_column("Tokens", style="green")
            step_table.add_column("Cost", style="red")
            step_table.add_column("Time", style="magenta")
            
            for step in self.metrics["steps"]:
                step_table.add_row(
                    str(step["step_number"]),
                    step["action_type"],
                    str(step["total_tokens"]),
                    f"${step['total_cost']:.4f}",
                    f"{step['total_time']:.2f}s"
                )
            
            self.console.print(step_table)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        return self.metrics


class BenchmarkHandler(ABC):
    """Abstract base class for benchmark-specific handlers.
    
    To add a new benchmark, simply create a new handler class that inherits from this
    or one of the specialized handlers (QAHandler, MathHandler, CodeHandler).
    """
    
    def __init__(self, llm: BaseLLM, max_steps: int = 6, testing: bool = False):
        self.llm = llm
        self.max_steps = max_steps
        self.testing = testing
    
    @abstractmethod
    def get_prompt(self) -> str:
        """Return the prompt template for this benchmark."""
        pass
    
    @abstractmethod
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse action string into action_type and query."""
        pass
    
    @abstractmethod
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle observation based on action type and query.
        
        Returns:
            Tuple of (observation_text, answer, finished, external_tool_info)
        """
        pass
    
    def format_action(self, action_type: str, query: str) -> str:
        """Format action for display in scratchpad."""
        return f"{action_type}[{query}]"
    
    def reset(self) -> None:
        """Reset internal state. Override if needed."""
        pass


class QAHandler(BenchmarkHandler):
    """Handler for QA benchmarks (HotpotQA, FEVER, TriviaQA, AmbigNQ).
    
    Supports Search, Lookup, and Finish actions with Wikipedia integration.
    """
    
    def __init__(self, llm: BaseLLM, max_steps: int = 6, testing: bool = False):
        super().__init__(llm, max_steps, testing)
        self.docstore = DocstoreExplorer(Wikipedia())
    
    def get_prompt(self) -> str:
        # This will be overridden by specific QA benchmarks
        return ""
    
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse QA action (Search, Lookup, Finish)."""
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, action)
        return (match.group(1), match.group(2)) if match else ("", "")
    
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle QA observation."""
        answer = ""
        finished = False
        external_tool_info = {}
        
        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = query
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)
            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        
        return obs, answer, finished, external_tool_info


class MathHandler(BenchmarkHandler):
    """Handler for Math benchmarks (GSM8K, SVAMP, TabMWP).
    
    Supports Calculate and Finish actions with Python code execution.
    """
    
    def get_prompt(self) -> str:
        # This will be overridden by specific Math benchmarks
        return ""
    
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse math action (Calculate, Finish)."""
        action_split = action.split("```python", maxsplit=1)
        match = re.search(r"\b(Finish|Calculate)\b", action_split[0], re.IGNORECASE)
        action_type = match.group(0).lower().capitalize() if match else ""
        try:
            query = action_split[1].split("```")[0].strip() if action_type else ""
        except:
            action_type = ""
            query = ""
        return action_type, query
    
    def format_action(self, action_type: str, query: str) -> str:
        """Format math action with code blocks."""
        return f"{action_type}[\n```python\n{query}\n```\n]"
    
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle math observation."""
        answer = ""
        finished = False
        external_tool_info = {}
        
        if action_type.lower() == "finish":
            answer = query
            finished = True
            obs = f"\n```python\n{answer}\n```"
        elif action_type.lower() == "calculate":
            code_answer, execution_status = safe_execute(query)
            external_tool_info["code_answer"] = code_answer[0]
            external_tool_info["execution_status"] = execution_status
            answer = query
            obs = f"\n```python\n{answer}\n```\nExecution Status: {execution_status}\nOutput: answer = {code_answer[0]}"
        else:
            obs = "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
        
        return obs, answer, finished, external_tool_info


class CodeHandler(BenchmarkHandler):
    """Handler for Code benchmarks (HumanEval, MBPP).
    
    Supports Implement, Test, and Finish actions with Python code execution.
    """
    
    def __init__(self, llm: BaseLLM, max_steps: int = 6, testing: bool = False):
        super().__init__(llm, max_steps, testing)
        self._answer = ""
    
    def get_prompt(self) -> str:
        # This will be overridden by specific Code benchmarks
        return ""
    
    def parse_action(self, action: str) -> Tuple[str, str]:
        """Parse code action (Implement, Test, Finish)."""
        action_split = action.split("```python", maxsplit=1)
        match = re.search(r"\b(Finish|Test|Implement)\b", action_split[0], re.IGNORECASE)
        action_type = match.group(0).lower().capitalize() if match else ""
        try:
            query = action_split[1].split("```")[0].strip() if action_type else ""
        except:
            action_type = ""
            query = ""
        return action_type, query
    
    def format_action(self, action_type: str, query: str) -> str:
        """Format code action with code blocks."""
        return f"{action_type}[\n```python\n{query}\n```\n]"
    
    def handle_observation(
        self, 
        action_type: str, 
        query: str, 
        scratchpad: str
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Handle code observation."""
        finished = False
        external_tool_info = {}
        
        if action_type.lower() == "finish":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status
            self._answer = query
            finished = True
            obs = f"\n```python\n{self._answer}\n```"
        elif action_type.lower() == "implement":
            _, execution_status = safe_execute(query)
            external_tool_info["execution_status"] = execution_status
            self._answer = query
            obs = f"\n```python\n{self._answer}\n```\nExecution Status: {execution_status}"
        elif action_type.lower() == "test":
            obs = f"{self._answer}\n\n{query}"
            _, execution_status = safe_execute(obs)
            external_tool_info["execution_status"] = execution_status
            obs = f"\n```python\n{obs}\n```\nExecution Status: {execution_status}"
        else:
            obs = "Invalid Action. Valid Actions are Implement[code] Test[code] and Finish[answer]."
        
        return obs, f"\n```python\n{self._answer}\n```\n", finished, external_tool_info
    
    def reset(self) -> None:
        """Reset internal state."""
        self._answer = ""


# Specific benchmark handlers - these are the "plugins"
class HotpotQAHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_HOTPOTQA


class FEVERHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_FEVER


class TriviaQAHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_TRIVIAQA


class AmbigNQHandler(QAHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_AMBIGNQ


class GSM8KHandler(MathHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_GSM8K


class SVAMPHandler(MathHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_SVAMP


class TabMWPHandler(MathHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_TABMWP


class HumanEvalHandler(CodeHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_HUMANEVAL


class MBPPHandler(CodeHandler):
    def get_prompt(self) -> str:
        return REACT_INSTRUCTION_MBPP


# Registry for benchmark handlers - this is the plugin registry
BENCHMARK_HANDLERS: Dict[str, Type[BenchmarkHandler]] = {
    Benchmarks.HOTPOTQA: HotpotQAHandler,
    Benchmarks.FEVER: FEVERHandler,
    Benchmarks.TRIVIAQA: TriviaQAHandler,
    Benchmarks.AMBIGNQ: AmbigNQHandler,
    Benchmarks.GSM8K: GSM8KHandler,
    Benchmarks.SVAMP: SVAMPHandler,
    Benchmarks.TABMWP: TabMWPHandler,
    Benchmarks.HUMANEVAL: HumanEvalHandler,
    Benchmarks.MBPP: MBPPHandler,
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
        
        if benchmark not in BENCHMARK_HANDLERS:
            available = list(BENCHMARK_HANDLERS.keys())
            raise ValueError(f"Unsupported benchmark: {benchmark}. Available: {available}")
            
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
        scratchpad += self.handler.format_action(action_type, query)
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
        # Start logging
        self.logger.log_start(self.benchmark, question)
        
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
                thought_response, action_response, finished
            )
            
            # Observe
            scratchpad, answer, obs, finished, external_tool_info = self._generate_observation(
                idx, scratchpad, action_type, query
            )
            
            # Log the observation
            self.logger.log_observation(obs, finished)
            
            steps.append(ReActStepOutput(
                thought=thought,
                action_type=action_type,
                query=query,
                observation=obs,
                answer=answer,
                external_tool_info=external_tool_info,
                thought_response=thought_response,
                action_response=action_response,
            ))
            
            idx += 1
        
        # Log completion
        self.logger.log_finish(answer)
        
        # Get metrics from logger
        metrics = self.logger.get_metrics()
        
        return ReActOutput(
            answer=answer,
            total_prompt_tokens=metrics["total_tokens"],  # Simplified for backward compatibility
            total_completion_tokens=0,  # Not tracked separately in new system
            total_tokens=metrics["total_tokens"],
            total_prompt_cost=metrics["total_cost"],  # Simplified for backward compatibility
            total_completion_cost=0,  # Not tracked separately in new system
            total_cost=metrics["total_cost"],
            total_prompt_time=metrics["total_time"],  # Simplified for backward compatibility
            total_time=metrics["total_time"],
            additional_info=steps,
        )
    
    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.handler.reset()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics from the logger."""
        return self.logger.get_metrics()
    
    @staticmethod
    def get_fewshots(benchmark: str, fewshot_type: str, **kwargs: Any) -> Dict[str, str]:
        """Get few-shot examples for the benchmark."""
        if benchmark not in BENCHMARK_HANDLERS:
            raise ValueError(f"Benchmark '{benchmark}' not found for ReAct.")
        
        if fewshot_type not in [FewShotType.REACT]:
            raise ValueError(f"Benchmark '{benchmark}' few-shot type not supported for ReAct.")
        
        benchmark_fewshots = BENCHMARK_FEWSHOTS[benchmark][fewshot_type]
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
        return list(BENCHMARK_HANDLERS.keys()) 