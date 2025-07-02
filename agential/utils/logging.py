"""Shared logging utilities for agents."""

from typing import Any, Dict

from agential.core.llm import Response
from rich.console import Console


class AgentLogger:
    """Simple logging for agents with metrics tracking.

    This logger provides step-by-step logging with metrics tracking,
    Rich console output, and summary generation.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.console = Console() if verbose else None
        self.metrics = {
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
        }

    def log_step(
        self,
        step_number: int,
        thought: str,
        action_type: str,
        query: str,
        thought_response: Response,
        action_response: Response,
    ):
        """Log a single step with metrics."""
        # Update metrics
        step_tokens = (
            thought_response.prompt_tokens
            + thought_response.completion_tokens
            + action_response.prompt_tokens
            + action_response.completion_tokens
        )
        step_cost = (
            thought_response.prompt_cost
            + thought_response.completion_cost
            + action_response.prompt_cost
            + action_response.completion_cost
        )

        self.metrics["total_steps"] += 1
        self.metrics["total_tokens"] += step_tokens
        self.metrics["total_cost"] += step_cost

        if self.verbose and self.console:
            self.console.print(
                f"Step {step_number}: {action_type}[{query[:50]}{'...' if len(query) > 50 else ''}]"
            )

    def log_finish(self, answer: str):
        """Log the completion of agent execution."""
        if self.verbose and self.console:
            self.console.print(f"🎯 Answer: {answer}")
            self.console.print(
                f"📊 Steps: {self.metrics['total_steps']}, Tokens: {self.metrics['total_tokens']}, Cost: ${self.metrics['total_cost']:.4f}"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset the logger metrics."""
        self.metrics = {
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_time": 0.0,
        }
