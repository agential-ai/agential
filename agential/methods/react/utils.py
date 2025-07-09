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


def parse_thought(text: str) -> str:
    """
    Parse thought from text, handling various formats.

    Handles:
    - Thought 1: <content>
    - Thought: <content>
    - <content> (no prefix)
    - Multi-line thoughts (splits at Action/Observation)

    Args:
        text (str): Raw thought text

    Returns:
        str: Cleaned thought content
    """
    text = text.split("Action")[0].strip()

    # Remove Thought X: prefix if present using string operations
    if ":" in text:
        # Split by first colon and take the part after it
        parts = text.split(":", 1)
        if len(parts) > 1:
            text = parts[1].strip()
    
    # Split at Action and take the first part
    return text


def parse_action(text: str, benchmark_type: str = "qa") -> Tuple[str, str]:
    """
    Parse action from text, handling various formats based on benchmark type.

    Args:
        text (str): Raw action text
        benchmark_type (str): Type of benchmark ("qa", "math", "code")

    Returns:
        Tuple[str, str]: (action_type, query)
    """
    text = text.strip()

    # Remove Action X: prefix if present
    action = re.sub(r"^Action \d+:\s*", "", text)
    action = re.sub(r"^Action:\s*", "", action)

    # Split at Observation and keep only the action part
    lines = action.splitlines()
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("Observation"):
            break
        if line_stripped:  # Only add non-empty lines
            filtered_lines.append(line)  # Keep original line with indentation

    action = "\n".join(filtered_lines).strip()

    if benchmark_type == "qa":
        return _parse_qa_action(action)
    elif benchmark_type == "math":
        return _parse_math_action(action)
    elif benchmark_type == "code":
        return _parse_code_action(action)
    else:
        # Default to QA parsing
        return _parse_qa_action(action)


def _parse_qa_action(action: str) -> Tuple[str, str]:
    """
    Parse QA action (Search, Lookup, Finish).

    Args:
        action (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    # Extract action part from text that might contain both thought and action
    if "Action" in action:
        # Split by "Action" and take the last part (in case there are multiple "Action" mentions)
        action_parts = action.split("Action")
        if len(action_parts) > 1:
            action_text = action_parts[-1].strip()
        else:
            action_text = action
    else:
        action_text = action

    # Remove Action X: prefix if present using string operations
    if ":" in action_text:
        # Split by first colon and take the part after it
        parts = action_text.split(":", 1)
        if len(parts) > 1:
            action_text = parts[1].strip()
        else:
            action_text = action_text.strip()

    # Split at Observation and keep only the action part
    if "Observation" in action_text:
        action_text = action_text.split("Observation")[0].strip()

    # First try the standard format with closing bracket
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, action_text, re.DOTALL)

    if match:
        action_type = match.group(1)
        argument = match.group(2).strip()
        return action_type, argument
    
    # Try without closing bracket (common when action spans multiple lines)
    pattern_no_close = r"^(\w+)\[(.+)$"
    match = re.match(pattern_no_close, action_text, re.DOTALL)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2).strip()
        return action_type, argument
    
    # Fallback: return empty strings
    return "", ""


def _parse_math_action(action: str) -> Tuple[str, str]:
    """
    Parse math action (Calculate, Finish).

    Handles:
    - Calculate[code]
    - Calculate[
        ```python
        code
        ```
    ]
    - Finish[answer]

    Note: Preserves indentation for Python code blocks.
    """
    # Try Action[query] format first
    match = re.match(r"^(\w+)\[(.*)\]$", action, re.DOTALL)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
        return action_type, query

    # Try Action[query (no closing bracket)
    match = re.match(r"^(\w+)\[(.*)", action, re.DOTALL)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
        return action_type, query

    # Try code block format - be careful about indentation
    action_split = action.split("```python", maxsplit=1)
    if len(action_split) > 1:
        match = re.search(r"\b(Finish|Calculate)\b", action_split[0], re.IGNORECASE)
        if match:
            action_type = match.group(0).lower().capitalize()
            try:
                # Extract code between ```python and ```
                code_part = action_split[1]
                if "```" in code_part:
                    query = code_part.split("```")[0]
                    # Preserve indentation by not stripping
                    return action_type, query
                else:
                    # No closing ```, take the rest
                    return action_type, code_part
            except:
                pass

    # Fallback to QA parsing
    return _parse_qa_action(action)


def _parse_code_action(action: str) -> Tuple[str, str]:
    """
    Parse code action (Implement, Test, Finish).

    Handles:
    - Implement[code]
    - Implement[
        ```python
        code
        ```
    ]
    - Test[code]
    - Finish[answer]

    Note: Preserves indentation for Python code blocks.
    """
    # Try Action[query] format first
    match = re.match(r"^(\w+)\[(.*)\]$", action, re.DOTALL)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
        return action_type, query

    # Try Action[query (no closing bracket)
    match = re.match(r"^(\w+)\[(.*)", action, re.DOTALL)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
        return action_type, query

    # Try code block format - be careful about indentation
    action_split = action.split("```python", maxsplit=1)
    if len(action_split) > 1:
        match = re.search(
            r"\b(Finish|Test|Implement)\b", action_split[0], re.IGNORECASE
        )
        if match:
            action_type = match.group(0).lower().capitalize()
            try:
                # Extract code between ```python and ```
                code_part = action_split[1]
                if "```" in code_part:
                    query = code_part.split("```")[0]
                    # Preserve indentation by not stripping
                    return action_type, query
                else:
                    # No closing ```, take the rest
                    return action_type, code_part
            except:
                pass

    # Fallback to QA parsing
    return _parse_qa_action(action)


def parse_answer(answer: str) -> str:
    """
    Parse answer and format it appropriately.
    
    If the answer is a single number/float, format it as a Python assignment.
    Otherwise, return the answer as-is.
    
    Args:
        answer (str): The raw answer text
        
    Returns:
        str: Formatted answer
    """
    
    try:
        answer = answer.strip()
        float_val = float(answer)
        return f"\n```python\nanswer = {float_val}\n```\n"
    except:
        return f"\n```python\n{answer}\n```\n"
