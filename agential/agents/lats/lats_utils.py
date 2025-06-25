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
        print(f"OUTPUT: {parsed_output if parsed_output is not None else response.output_text}")
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


def _build_failed_trajectory_format(question: str, trajectory: str, reflection: str) -> str:
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


# def get_node_trajectory(node: Node) -> str:
#     """Get the trajectory string from a node in chronological order (root to leaf)."""
#     if not node or not node.state:
#         return ""

#     node_steps = []
#     current_node = node
#     while current_node and current_node.state:
#         state = current_node.state
#         step_parts = []
#         if 'thought' in state and state['thought']:
#             step_parts.append(f"Thought: {state['thought']}")
#         if 'action_type' in state and state['action_type'] and 'query' in state and state['query']:
#             step_parts.append(f"Action: {state['action_type']}[{state['query']}]")
#         if 'observation' in state and state['observation']:
#             step_parts.append(f"Observation: {state['observation']}")
#         node_steps.append(step_parts)
#         current_node = current_node.parent
#     return "\n".join(reversed(node_steps))


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
            if 'thought' in node.state and node.state['thought']:
                step.append(f"Thought {node.depth}: {node.state['thought']}")
            # if (
            #     'action_type' in node.state and node.state['action_type']
            #     and 'query' in node.state and node.state['query']
            # ):
            step.append(
                f"Action {node.depth}: {node.state['action_type']}[{node.state['query']}]"
            )
            if 'observation' in node.state and node.state['observation']:
                step.append(f"Observation {node.depth}: {node.state['observation']}")
        step_str = "\n".join(step)
        trajectory.append(step_str)
        node = node.parent  # type: ignore

    return "\n".join(reversed(trajectory))
    

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
    calculate_match = re.search(r"Calculate\[(.*?)\]", action, re.IGNORECASE)
    if calculate_match:
        return "Calculate", calculate_match.group(1).strip()

    finish_match = re.search(r"Finish\[(.*?)\]", action, re.IGNORECASE)
    if finish_match:
        return "Finish", finish_match.group(1).strip()

    # If no pattern matches, try to extract the first word as action type
    words = action.split()
    if words:
        action_type = words[0]
        query = " ".join(words[1:]) if len(words) > 1 else ""
        return action_type, query

    return "", "" 