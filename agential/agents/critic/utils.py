"""Shared utilities for CRITIC agents."""

from typing import Tuple
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

from agential.core.llm import Response


def log_llm_io(
    response: Response, 
    context: str = "", 
    verbose: bool = False, 
    truncate_length: int = -1
):
    """Log LLM input/output with rich formatting.

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


def remove_comment(code: str) -> str:
    """Removes all comment lines and empty lines from the given block of code.

    Args:
        code: A string containing the block of code from which comments and empty lines will be removed.

    Returns:
        The code with all comment lines that start with '#' and empty lines removed.
    """
    code_lines = code.split("\n")
    code_lines = [line for line in code_lines if not line.startswith("#")]
    code_lines = [line for line in code_lines if line.strip() != ""]
    return "\n".join(code_lines)


def parse_search_query(critique: str) -> Tuple[bool, str]:
    """Parse search query from critique text.
    
    Args:
        critique: The critique text to parse
        
    Returns:
        Tuple of (has_search_query, search_query)
    """
    if "> Search Query: " in critique:
        parts = critique.split("> Search Query:")
        if len(parts) >= 2:
            search_query = parts[1].split("\n")[0].strip()
            return True, search_query
    return False, ""


def parse_final_answer(critique: str) -> str:
    """Parse final answer from critique text.
    
    Args:
        critique: The critique text to parse
        
    Returns:
        The final answer
    """
    if "Answer: " in critique:
        return critique.split("Answer: ")[-1].strip()
    return critique.strip() 