"""Reflexion Agent (handler-based, modern version)."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from rich.console import Console

from agential.agents.base.agent import BaseAgent
from agential.core.llm import BaseLLM, Response
from agential.agents.reflexion.handlers import BENCHMARK_HANDLERS

# =============================================================================
# OUTPUT CLASSES
# =============================================================================
@dataclass
class ReflexionStepOutput:
    thought: str
    action_type: str
    query: str
    observation: str
    answer: str
    reflection: str = ""
    memory: str = ""
    raw_thought: str = ""
    raw_action: str = ""
    raw_reflection: str = ""
    raw_memory: str = ""
    external_tool_info: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ReflexionOutput:
    answer: str
    total_tokens: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    steps: List[ReflexionStepOutput] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    memories: List[str] = field(default_factory=list)
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    def summary(self) -> Dict[str, Any]:
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
    def __init__(self, verbose: bool = True, truncate_length: int = 200, debug_mode: bool = False):
        self.verbose = verbose
        self.console = Console() if verbose else None
        self.truncate_length = truncate_length
        self.debug_mode = debug_mode
        self.metrics = {
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
        }
    def _truncate_text(self, text: str, prefix: str = "") -> str:
        if len(text) <= self.truncate_length:
            return text
        truncated = text[:self.truncate_length].rstrip()
        return f"{truncated}{prefix}... (truncated)"
    def log_step(self, step_number: int, thought: str, action_type: str, query: str, 
                 thought_response: Response, action_response: Response, 
                 raw_thought: str = "", raw_action: str = ""):
        step_tokens = (thought_response.prompt_tokens + thought_response.completion_tokens + 
                      action_response.prompt_tokens + action_response.completion_tokens)
        step_cost = (thought_response.prompt_cost + thought_response.completion_cost + 
                    action_response.prompt_cost + action_response.completion_cost)
        self.metrics["total_steps"] += 1
        self.metrics["total_tokens"] += step_tokens
        self.metrics["total_cost"] += step_cost
        if self.verbose and self.console:
            self.console.print(f"\n[bold blue]Step {step_number}[/bold blue]")
            self.console.print("─" * 50)
            truncated_thought = self._truncate_text(thought)
            self.console.print(f"[bold green]💭 Thought:[/bold green]")
            self.console.print(f"   {truncated_thought}")
            if self.debug_mode and raw_thought:
                self.console.print(f"[bold red]🐛 DEBUG - Raw LLM Thought Response:[/bold red]")
                self.console.print(f"   {raw_thought}")
            truncated_query = self._truncate_text(query, " (truncated)")
            self.console.print(f"[bold yellow]🔧 Action:[/bold yellow] {action_type}[{truncated_query}]")
            if self.debug_mode or (action_type == "" and raw_action):
                self.console.print(f"[bold red]🐛 DEBUG - Raw LLM Action Response:[/bold red]")
                self.console.print(f"   {raw_action}")
                if action_type == "":
                    self.console.print(f"[bold red]   ⚠️  Action parsing failed![/bold red]")
            self.console.print(f"[dim]📊 Step tokens: {step_tokens}, Step cost: ${step_cost:.4f}[/dim]")
    def log_observation(self, step_number: int, observation: str, answer: str, finished: bool, 
                       action_type: str = "", query: str = ""):
        if self.verbose and self.console:
            truncated_obs = self._truncate_text(observation)
            truncated_answer = self._truncate_text(answer) if answer else ""
            self.console.print(f"[bold magenta]👁️  Observation:[/bold magenta]")
            self.console.print(f"   {truncated_obs}")
            if "Invalid Action" in observation:
                self.console.print(f"[bold red]🚨 Invalid Action Detected![/bold red]")
                self.console.print(f"   Action Type: '{action_type}'")
                self.console.print(f"   Query: '{query}'")
                self.console.print(f"   [dim]Check the raw LLM response above for debugging[/dim]")
            if finished:
                self.console.print(f"[bold green]✅ Finished![/bold green]")
                if truncated_answer:
                    self.console.print(f"[bold green]   Answer: {truncated_answer}[/bold green]")
            else:
                self.console.print(f"[dim]⏭️  Continuing to next step...[/dim]")
            self.console.print("─" * 50)
    def log_reflection(self, reflection: str, raw_reflection: str = ""):
        if self.verbose and self.console:
            self.console.print(f"[bold cyan]🔄 Reflection:[/bold cyan] {self._truncate_text(reflection)}")
            if self.debug_mode and raw_reflection:
                self.console.print(f"[bold red]🐛 DEBUG - Raw LLM Reflection Response:[/bold red]")
                self.console.print(f"   {raw_reflection}")
    def log_memory(self, memory: str, raw_memory: str = ""):
        if self.verbose and self.console:
            self.console.print(f"[bold white]🧠 Memory Update:[/bold white] {self._truncate_text(memory)}")
            if self.debug_mode and raw_memory:
                self.console.print(f"[bold red]🐛 DEBUG - Raw LLM Memory Response:[/bold red]")
                self.console.print(f"   {raw_memory}")
    def log_finish(self, answer: str):
        if self.verbose and self.console:
            truncated_answer = self._truncate_text(answer)
            self.console.print(f"\n[bold green]🎯 Final Answer:[/bold green] {truncated_answer}")
            self.console.print(f"[bold blue]📊 Summary:[/bold blue] Steps: {self.metrics['total_steps']}, Tokens: {self.metrics['total_tokens']}, Cost: ${self.metrics['total_cost']:.4f}")
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

# =============================================================================
# MAIN REFLEXION AGENT
# =============================================================================
class ReflexionAgent(BaseAgent):
    """Reflexion agent with handler-based architecture and enhanced logging.
    Supports debug mode for raw LLM output logging and unified output dataclasses.
    """
    def __init__(
        self,
        llm: BaseLLM,
        benchmark: str,
        max_steps: int = 6,
        max_reflections: int = 3,
        verbose: bool = True,
        truncate_length: int = 200,
        debug_mode: bool = False,
        testing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, benchmark=benchmark, testing=testing)
        if benchmark not in BENCHMARK_HANDLERS:
            raise ValueError(f"Handler not found for benchmark: {benchmark}")
        handler_class = BENCHMARK_HANDLERS[benchmark]
        self.handler = handler_class(llm=llm, max_steps=max_steps, max_reflections=max_reflections, testing=testing)
        self.max_steps = max_steps
        self.max_reflections = max_reflections
        self.logger = AgentLogger(verbose=verbose, truncate_length=truncate_length, debug_mode=debug_mode)
    def generate(
        self,
        question: str,
        key: str = "",
        examples: str = "",
        prompt: str = "",
        reflect_examples: str = "",
        reflect_prompt: str = "",
        reflect_strategy: str = "reflexion",
        additional_keys: Dict[str, str] = {},
        reflect_additional_keys: Dict[str, str] = {},
        fewshot_type: str = "",
        patience: int = 3,
        reset: bool = True,
    ) -> ReflexionOutput:
        if reset:
            self.reset()
        scratchpad = ""
        answer = ""
        finished = False
        idx = 1
        steps = []
        reflections = []
        memories = []
        while not finished and idx <= self.max_steps:
            # Thought
            scratchpad += f"\nThought {idx}: "
            thought_response = self.handler.llm(scratchpad)
            raw_thought = thought_response.output_text
            thought = raw_thought.split("Action")[0].strip()
            scratchpad += thought
            # Action
            scratchpad += f"\nAction {idx}: "
            action_response = self.handler.llm(scratchpad)
            raw_action = action_response.output_text
            action = raw_action.split("Observation")[0]
            action_type, query = self.handler.parse_action(action)
            scratchpad += f"{action_type}[{query}]"
            # Log step
            self.logger.log_step(idx, thought, action_type, query, thought_response, action_response, raw_thought, raw_action)
            # Observation
            scratchpad += f"\nObservation {idx}: "
            obs, answer, finished, external_tool_info = self.handler.handle_observation(action_type, query, scratchpad)
            scratchpad += obs
            self.logger.log_observation(idx, obs, answer, finished, action_type, query)
            # Reflection (if applicable)
            reflection = ""
            raw_reflection = ""
            if reflect_strategy:
                reflections_list, reflection, reflection_response = self.handler.handle_reflection(
                    scratchpad, reflect_strategy, question, examples, reflect_prompt, reflect_additional_keys
                )
                reflections.append(reflection)
                raw_reflection = reflection_response.output_text if reflection_response else ""
                self.logger.log_reflection(reflection, raw_reflection)
            # Memory (if applicable)
            memory = ""
            raw_memory = ""
            # (Memory logic can be added here if needed)
            # self.logger.log_memory(memory, raw_memory)
            steps.append(ReflexionStepOutput(
                thought=thought,
                action_type=action_type,
                query=query,
                observation=obs,
                answer=answer,
                reflection=reflection,
                memory=memory,
                raw_thought=raw_thought,
                raw_action=raw_action,
                raw_reflection=raw_reflection,
                raw_memory=raw_memory,
                external_tool_info=external_tool_info,
            ))
            idx += 1
        self.logger.log_finish(answer)
        metrics = self.logger.get_metrics()
        return ReflexionOutput(
            answer=answer,
            total_tokens=metrics["total_tokens"],
            total_cost=metrics["total_cost"],
            total_time=metrics["total_time"],
            steps=steps,
            reflections=reflections,
            memories=memories,
        )
    def reset(self) -> None:
        self.handler.reset()
    def get_metrics(self) -> Dict[str, Any]:
        return self.logger.get_metrics()
    @staticmethod
    def list_benchmarks() -> List[str]:
        return list(BENCHMARK_HANDLERS.keys())
    @staticmethod
    def get_fewshots(benchmark: str, fewshot_type: str = "reflexion", **kwargs: Any) -> Dict[str, str]:
        # (Implement fewshot retrieval logic as needed)
        return {}
    @staticmethod
    def get_prompts(benchmark: str, **kwargs: Any) -> Dict[str, str]:
        # (Implement prompt retrieval logic as needed)
        return {}