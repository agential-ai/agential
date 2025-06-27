"""
Shared utilities for react agents.
"""

import re
from typing import Tuple
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape


def log_llm_io(
    response, context: str = "", verbose: bool = False, truncate_length: int = -1
):
    """
    Log LLM input/output with rich formatting.

    Args:
        response: LLM response object
        context: Context string for the log
        verbose: Whether to log
        truncate_length: Maximum length before truncating
    """
    if not verbose:
        return

    console = Console()
    input_text = escape(str(response.input_text))
    output_text = escape(response.output_text)

    if truncate_length != -1 and truncate_length > 0:
        if len(input_text) > truncate_length:
            input_text = input_text[:truncate_length] + "..."
        if len(output_text) > truncate_length:
            output_text = output_text[:truncate_length] + "..."

    # Escape context for rich markup
    context_escaped = escape(context)
    content = f"[bold blue]LLM {context_escaped}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{output_text}"
    console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))


def parse_thought_action(text: str) -> Tuple[str, str]:
    """
    General parser for Thought/Action blocks supporting single-line and multi-line formats.
    Handles:
      - Action[<query>]
      - Action[
          <multi-line query>
        ]
      - Action <query>
      - Action\n<query>
    Returns (action_type, query)
    Strips any trailing lines that start with 'Observation', 'Thought', or 'Action' (for the next step).
    """
    text = text.strip()
    # Action[ ... ] (single or multi-line)
    match = re.match(r"^(\w+)\[(.*)\]$", text, re.DOTALL)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
    else:
        # Action[ ... (no closing bracket, multi-line)
        match = re.match(r"^(\w+)\[(.*)", text, re.DOTALL)
        if match:
            action_type, query = match.group(1), match.group(2).strip()
        else:
            # Action\n<query> (multi-line, no brackets)
            match = re.match(r"^(\w+)\s*\n([\s\S]+)", text)
            if match:
                action_type, query = match.group(1), match.group(2).strip()
            else:
                # Action <query> (single line)
                match = re.match(r"^(\w+)\s+(.+)$", text)
                if match:
                    action_type, query = match.group(1), match.group(2).strip()
                else:
                    # Fallback: just the action type
                    action_type = text.split()[0] if text else ""
                    query = text[len(action_type):].strip() if action_type else ""
    # Remove any trailing lines that start with a prompt marker
    if query:
        lines = query.splitlines()
        filtered = []
        for line in lines:
            if line.strip().startswith(("Observation", "Thought", "Action")):
                break
            filtered.append(line)
        query = "\n".join(filtered).strip()
    return action_type, query
