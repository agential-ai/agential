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
    text = text.strip()
    
    # Remove Thought X: prefix if present
    thought = re.sub(r"^Thought \d+:\s*", "", text)
    thought = re.sub(r"^Thought:\s*", "", thought)
    
    # Split at Action or Observation and keep only the thought part
    lines = thought.splitlines()
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith(("Action", "Observation")):
            break
        if line_stripped:  # Only add non-empty lines
            filtered_lines.append(line)  # Keep original line with indentation
    
    return "\n".join(filtered_lines).strip()


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
    
    Handles:
    - Action[query]
    - Action <query>
    - Action\n<query>
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
    
    # Try Action\n<query> format
    match = re.match(r"^(\w+)\s*\n([\s\S]+)", action)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
        return action_type, query
    
    # Try Action <query> format
    match = re.match(r"^(\w+)\s+(.+)$", action)
    if match:
        action_type, query = match.group(1), match.group(2).strip()
        return action_type, query
    
    # Fallback
    action_type = action.split()[0] if action else ""
    query = action[len(action_type):].strip() if action_type else ""
    return action_type, query


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
        match = re.search(r"\b(Finish|Test|Implement)\b", action_split[0], re.IGNORECASE)
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