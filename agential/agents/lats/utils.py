"""
Shared utilities for LATS agents.
"""

import re
from typing import Dict, Tuple, Optional
from rich.console import Console
from agential.core.llm import BaseLLM, Response
from agential.agents.lats.node import Node
from rich.panel import Panel
from rich.markup import escape

console = Console()


def log_llm_io(
    response,
    context: str = "",
    verbose: bool = False,
    truncate_length: int = -1,
    parsed_output: Optional[str] = None,
    depth: Optional[int] = None,
    node_index: Optional[int] = None,
    retry: Optional[int] = None,
    extra_info: Optional[str] = None,
):
    """Log LLM input/output with rich formatting and detailed context."""
    if not verbose:
        return

    try:
        input_text = escape(str(response.input_text))
        output_text = escape(response.output_text)
        display_output = parsed_output if parsed_output is not None else output_text

        # Compose detailed context
        context_parts = [context]
        if depth is not None:
            context_parts.append(f"Depth: {depth}")
        if node_index is not None:
            context_parts.append(f"Node: {node_index}")
        if retry is not None and retry > 0:
            context_parts.append(f"Retry: {retry}")
        if extra_info:
            context_parts.append(str(extra_info))
        context_escaped = escape(" | ".join(context_parts))

        if truncate_length != -1 and truncate_length > 0:
            if len(input_text) > truncate_length:
                input_text = input_text[:truncate_length] + "..."
            if len(display_output) > truncate_length:
                display_output = display_output[:truncate_length] + "..."

        content = f"[bold blue]LLM {context_escaped}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{display_output}"
        console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))
    except ImportError:
        # Fallback to simple logging if rich is not available
        print(f"LLM {context}:")
        print(f"INPUT: {response.input_text}")
        print(
            f"OUTPUT: {parsed_output if parsed_output is not None else response.output_text}"
        )
        print("-" * 50)


# Format strings for LATS
LATS_REFLECTION_FORMAT = """{trajectory}

Reflection: {reflection}"""

LATS_FAILED_TRAJECTORY_FORMAT = """Question: {question}
{trajectory}

Explanation: This trajectory is incorrect as {reflection}
Correctness score: 1"""


def _build_reflection_format(trajectory: str, reflection: str) -> str:
    """Builds a formatted string for LATS reflection."""
    return LATS_REFLECTION_FORMAT.format(trajectory=trajectory, reflection=reflection)


def _build_failed_trajectory_format(
    question: str, trajectory: str, reflection: str
) -> str:
    """Builds a formatted string for a failed LATS trajectory."""
    return LATS_FAILED_TRAJECTORY_FORMAT.format(
        question=question, trajectory=trajectory, reflection=reflection
    )


def _build_agent_prompt(
    question: str,
    examples: str,
    trajectory: str,
    reflections: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs an agent prompt for the LATS agent."""
    prompt_kwargs = {
        "question": question,
        "examples": examples,
        "trajectory": trajectory,
        "reflections": reflections,
        **additional_keys,
    }
    return prompt.format(**prompt_kwargs)


def _prompt_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    trajectory: str,
    reflections: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates an agent response using the language model."""
    full_prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        reflections=reflections,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    return llm(full_prompt)


def _build_value_prompt(
    question: str,
    examples: str,
    trajectory: str,
    failed_trajectories: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a value prompt for the LATS agent."""
    prompt_kwargs = {
        "question": question,
        "examples": examples,
        "trajectory": trajectory,
        "failed_trajectories": failed_trajectories,
        **additional_keys,
    }
    return prompt.format(**prompt_kwargs)


def _prompt_value(
    llm: BaseLLM,
    question: str,
    examples: str,
    trajectory: str,
    failed_trajectories: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a value assessment using the language model."""
    full_prompt = _build_value_prompt(
        question=question,
        examples=examples,
        trajectory=trajectory,
        failed_trajectories=failed_trajectories,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    return llm(full_prompt)


def get_node_trajectory(node: Node) -> str:
    """Generates a string representation of the trajectory from the given node to the root.

    Args:
        node (Node): The current node in the tree.

    Returns:
        str: A string representation of the trajectory, including thoughts, actions, and observations.
    """
    trajectory = []

    while node:
        step = []
        if node.depth > 0:
            if "thought" in node.state and node.state["thought"]:
                step.append(f"Thought {node.depth}: {node.state['thought']}")
            step.append(
                f"Action {node.depth}: {node.state['action_type']}[{node.state['query']}]"
            )
            if "observation" in node.state and node.state["observation"]:
                step.append(f"Observation {node.depth}: {node.state['observation']}")
        step_str = "\n".join(step)
        trajectory.append(step_str)
        node = node.parent  # type: ignore

    return "\n".join(reversed(trajectory))


def clean_llm_output(text: str) -> str:
    """Clean LLM output by removing step prefixes like 'Thought 1:' or 'Action 5:'.
    
    Args:
        text (str): The raw LLM output text
        
    Returns:
        str: The cleaned text without step prefixes
    """
    # Remove common step prefixes
    prefixes_to_remove = [
        r"^Thought\s+\d+:\s*",
        r"^Action\s+\d+:\s*", 
        r"^Observation\s+\d+:\s*",
    ]
    
    cleaned_text = text.strip()
    for prefix_pattern in prefixes_to_remove:
        cleaned_text = re.sub(prefix_pattern, "", cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text.strip()


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================


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
    """Parse a QA action string to extract action type and query."""
    # Remove any leading/trailing whitespace and newlines
    string = string.strip()

    # Look for patterns like "Search[query]", "Lookup[query]", "Finish[answer]"
    search_match = re.search(r"Search\[(.*?)\]", string, re.IGNORECASE)
    if search_match:
        return "Search", search_match.group(1).strip()

    lookup_match = re.search(r"Lookup\[(.*?)\]", string, re.IGNORECASE)
    if lookup_match:
        return "Lookup", lookup_match.group(1).strip()

    finish_match = re.search(r"Finish\[(.*?)\]", string, re.IGNORECASE)
    if finish_match:
        return "Finish", finish_match.group(1).strip()

    # If no pattern matches, try to extract the first word as action type
    words = string.split()
    if words:
        action_type = words[0]
        query = " ".join(words[1:]) if len(words) > 1 else ""
        return action_type, query

    return "", ""


def parse_math_action(action: str) -> Tuple[str, str]:
    """Parse a math action string to extract action type and query."""
    # Remove any leading/trailing whitespace and newlines
    action = action.strip()

    # Look for patterns like "Calculate[expression]", "Finish[answer]"
    # Use DOTALL flag to handle multiline content
    calculate_match = re.search(r"Calculate\[(.*?)\]", action, re.IGNORECASE | re.DOTALL)
    if calculate_match:
        return "Calculate", calculate_match.group(1).strip()

    finish_match = re.search(r"Finish\[(.*?)\]", action, re.IGNORECASE | re.DOTALL)
    if finish_match:
        return "Finish", finish_match.group(1).strip()

    # If no pattern matches, try to extract the first word as action type
    words = action.split()
    if words:
        action_type = words[0]
        query = " ".join(words[1:]) if len(words) > 1 else ""
        return action_type, query

    return "", ""


def parse_code_action(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`, `Test`, `Implement`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    # Remove any leading/trailing whitespace and newlines
    action = action.strip()

    # Look for patterns like "Implement[```python ... ```]", "Test[```python ... ```]", "Finish[```python ... ```]"
    # Use DOTALL flag to handle multiline content
    implement_match = re.search(r"Implement\[\s*```python(.*?)```\s*\]", action, re.IGNORECASE | re.DOTALL)
    if implement_match:
        return "Implement", implement_match.group(1).strip()

    test_match = re.search(r"Test\[\s*```python(.*?)```\s*\]", action, re.IGNORECASE | re.DOTALL)
    if test_match:
        return "Test", test_match.group(1).strip()

    finish_match = re.search(r"Finish\[\s*```python(.*?)```\s*\]", action, re.IGNORECASE | re.DOTALL)
    if finish_match:
        return "Finish", finish_match.group(1).strip()

    # Fallback: try to extract action type and code without strict formatting
    action_type_match = re.search(r"\b(Finish|Test|Implement)\b", action, re.IGNORECASE)
    if action_type_match:
        action_type = action_type_match.group(0).lower().capitalize()
        # Try to extract code after the action type
        code_match = re.search(r"```python(.*?)```", action, re.DOTALL)
        if code_match:
            query = code_match.group(1).strip()
        else:
            # If no code blocks found, try to extract everything after the action type
            action_parts = action.split(action_type_match.group(0), 1)
            if len(action_parts) > 1:
                query = action_parts[1].strip()
                # Remove brackets if present
                query = re.sub(r"^\[(.*)\]$", r"\1", query, flags=re.DOTALL)
            else:
                query = ""
        return action_type, query

    # If no pattern matches, try to extract the first word as action type
    words = action.split()
    if words:
        action_type = words[0]
        query = " ".join(words[1:]) if len(words) > 1 else ""
        return action_type, query

    return "", ""


def parse_latest_implement(text: str) -> str:
    """Extract the latest Python code implementation from the given text.

    This function searches for the last occurrence of Python code enclosed in
    'Implement[```python ... ```]' blocks within the input text.

    Args:
        text (str): The input text containing one or more code implementations.

    Returns:
        str: The extracted Python code as a string if found, or "" if no implementation is found.
    """
    pattern = re.compile(r"Implement\[\s*```python(.*?)```", re.DOTALL)

    matches = pattern.findall(text)

    if matches:
        latest_implement = matches[-1].strip()
        return latest_implement
    return ""


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
