"""
Shared utilities for lats agents.
"""

import re
from typing import Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape


def parse_llm_response(response_text: str) -> Tuple[str, str, str]:
    """
    Parse LLM response to extract thought, action_type, and query.

    Args:
        response_text: The raw text response from the LLM

    Returns:
        Tuple of (thought, action_type, query) where each can be empty string if parsing fails
    """
    # Primary parsing with strict regex
    thought_match = re.search(r"Thought.*?:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    action_match = re.search(
        r"Action.*?:\s*([\w]+)\[(.*?)]\s*(?:\n|$)", response_text, re.DOTALL
    )

    thought = thought_match.group(1).strip() if thought_match else ""
    action_type = action_match.group(1) if action_match else ""
    query = action_match.group(2).strip() if action_match else ""

    # Fallback parsing if primary parsing failed
    if not thought:
        thought_match = re.search(r"Thought.*?:\s*(.*)", response_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

    if not action_type:
        action_fallback = re.search(r"Action.*?:\s*(.*)", response_text, re.DOTALL)
        action_raw = action_fallback.group(1).strip() if action_fallback else ""

        # Try to parse action with various patterns
        match = re.match(r"^(\w+)\[(.*)\]$", action_raw.strip(), re.DOTALL)
        if match:
            action_type = match.group(1)
            query = match.group(2).strip()
        else:
            match = re.match(r"^(\w+)\[(.*)", action_raw.strip(), re.DOTALL)
            if match:
                action_type = match.group(1)
                query = match.group(2).strip()
            else:
                # Last resort: split on whitespace
                action_type = (
                    action_raw.strip().split()[0] if action_raw.strip() else ""
                )
                query = action_raw.strip()[len(action_type) :].strip()

    return thought, action_type, query


def parse_action_string(action_string: str) -> Tuple[str, str]:
    """
    Parse action string to extract action_type and query.

    Args:
        action_string: String in format "action_type[query]" or similar

    Returns:
        Tuple of (action_type, query)
    """
    # Try to parse with various patterns
    match = re.match(r"^(\w+)\[(.*)\]$", action_string.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()

    match = re.match(r"^(\w+)\[(.*)", action_string.strip(), re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()

    # Last resort: split on whitespace
    action_type = action_string.strip().split()[0] if action_string.strip() else ""
    query = action_string.strip()[len(action_type) :].strip()
    return action_type, query


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


def parse_value(string: str) -> Tuple[str, float]:
    """Extracts the explanation and correctness score from a given string.

    Args:
        string (str): The input string containing an explanation and correctness score.

    Returns:
        Tuple[str, float]: A tuple containing the explanation (str) and the correctness score (float).
        If parsing fails, returns ("Explanation not found", 0.0).
    """
    try:
        explanation_part = string.split("Explanation:")[1].strip()
        explanation, score_part = explanation_part.split("Correctness score:")
        score = float(int(score_part.strip()))
        return explanation.strip(), score
    except Exception:
        return "Explanation not found", 0.0


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

    if truncate_length is not None:
        if len(input_text) > truncate_length:
            input_text = input_text[:truncate_length] + "..."
        if len(output_text) > truncate_length:
            output_text = output_text[:truncate_length] + "..."

    # Escape context for rich markup
    context_escaped = escape(context)
    content = f"[bold blue]LLM {context_escaped}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{output_text}"
    console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))
