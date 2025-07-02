"""
Shared utilities for EXPEL agents.
"""

import random
import re
from typing import Any, Dict, List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

from agential.methods.expel.prompts import (
    CRITIQUE_SUMMARY_SUFFIX_FULL,
    CRITIQUE_SUMMARY_SUFFIX_NOT_FULL,
    EXISTING_INSIGHTS_AI_NAME,
    HUMAN_CRITIQUE_EXISTING_INSIGHTS_ALL_SUCCESS_TEMPLATE,
    HUMAN_CRITIQUE_EXISTING_INSIGHTS_TEMPLATE,
    NON_EXISTENT_INSIGHTS_AT_NAME,
    SYSTEM_CRITIQUE_EXISTING_INSIGHTS_INSTRUCTION,
    SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_INSIGHTS_INSTRUCTION,
    SYSTEM_TEMPLATE,
)

console = Console()


# ============================================================================
# INSIGHT EXTRACTION
# ============================================================================


def _build_compare_prompt(
    insights: List[Dict[str, Any]],
    question: str,
    success_trial: str,
    failed_trial: str,
    is_full: bool,
) -> str:
    """Constructs a comparison prompt for an AI by combining system instructions, task details, and a list of existing insights.

    This function formats a prompt intended for AI to critique existing insights based on a given task. The task is described by a question and includes examples of both successful and failed trials.

    Parameters:
        insights (List[Dict[str, Any]]): A list of insight dictionaries.
        question (str): The question that defines the task.
        success_trial (str): A description or example of a successful trial for the task.
        failed_trial (str): A description or example of a failed trial for the task.
        is_full (bool): A flag indicating whether the prompt should be in its full form or not. This affects the suffix of the critique summary.

    Returns:
        str: A fully constructed prompt ready to be presented to the AI.
    """
    # System prompt.
    prefix = SYSTEM_TEMPLATE.format(
        ai_name=(
            NON_EXISTENT_INSIGHTS_AT_NAME if not insights else EXISTING_INSIGHTS_AI_NAME
        ),
        instruction=SYSTEM_CRITIQUE_EXISTING_INSIGHTS_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "question": question,
        "failed_traj": failed_trial,
        "success_traj": success_trial,
        "existing_insights": (
            "\n".join(
                [f"{i}. {insight['insight']}" for i, insight in enumerate(insights)]
            )
            if insights
            else ""
        ),
    }

    human_critique_summary_message = HUMAN_CRITIQUE_EXISTING_INSIGHTS_TEMPLATE.format(
        **human_format_dict
    )
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def _build_all_success_prompt(
    insights: List[Dict[str, Any]],
    success_trajs_str: str,
    is_full: bool,
) -> str:
    """Constructs a prompt for AI to critique existing insights based on all successful trajectories.

    This function formats a prompt intended for AI to critique existing insights when all trials are successful. The prompt includes system instructions, task details, and a list of existing insights.

    Parameters:
        insights (List[Dict[str, Any]]): A list of insight dictionaries.
        success_trajs_str (str): A string containing all successful trajectories.
        is_full (bool): A flag indicating whether the prompt should be in its full form or not. This affects the suffix of the critique summary.

    Returns:
        str: A fully constructed prompt ready to be presented to the AI.
    """
    # System prompt.
    prefix = SYSTEM_TEMPLATE.format(
        ai_name=(
            NON_EXISTENT_INSIGHTS_AT_NAME if not insights else EXISTING_INSIGHTS_AI_NAME
        ),
        instruction=SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_INSIGHTS_INSTRUCTION,
    )

    # Task prompt.
    human_format_dict = {
        "success_trajs": success_trajs_str,
        "existing_insights": (
            "\n".join(
                [f"{i}. {insight['insight']}" for i, insight in enumerate(insights)]
            )
            if insights
            else ""
        ),
    }

    human_critique_summary_message = (
        HUMAN_CRITIQUE_EXISTING_INSIGHTS_ALL_SUCCESS_TEMPLATE.format(
            **human_format_dict
        )
    )
    critique_summary_suffix = (
        CRITIQUE_SUMMARY_SUFFIX_FULL if is_full else CRITIQUE_SUMMARY_SUFFIX_NOT_FULL
    )

    prompt = prefix + "\n" + human_critique_summary_message + critique_summary_suffix

    return prompt


def parse_insights(llm_text: str) -> List[Tuple[str, str]]:
    """Parses and extracts insight operations and their descriptions from a given text.

    This function searches through the provided text for occurrences of insight operations (ADD, REMOVE, EDIT, AGREE) followed by their descriptions.
    It applies specific criteria to ensure the extracted insights are valid: the insight description must not be empty, must not
    contain certain banned words (to avoid inclusion of formatting instructions or similar), and must end with a period.

    Parameters:
        llm_text (str): The text from which to extract insight operations and descriptions.
            This text is expected to contain one or more statements formatted according to predefined insight operation patterns.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains two elements: the operation (ADD, REMOVE, EDIT, AGREE) and the clean, validated insight description.
            The insights that do not meet the validation criteria are omitted.
    """
    pattern = r"((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)"
    matches = re.findall(pattern, llm_text)

    res = []
    banned_words = ["ADD", "AGREE", "EDIT"]
    for operation, text in matches:
        text = text.strip()
        if (
            text != ""
            and not any([w in text for w in banned_words])
            and text.endswith(".")
        ):
            # If text is not empty.
            # If text doesn't contain banned words (avoid weird formatting cases from llm).
            # If text ends with a period (avoid cut off sentences from llm).
            if "ADD" in operation:
                res.append(("ADD", text))
            else:
                res.append((operation.strip(), text))
    return res


def retrieve_insight_index(
    insights: List[Dict[str, Any]], operation_rule_text: str
) -> int:
    """Retrieves the index of an insight based on operation rule text.

    This function searches through a list of insights to find one that matches the given operation rule text.

    Parameters:
        insights (List[Dict[str, Any]]): A list of insight dictionaries.
        operation_rule_text (str): The operation rule text to search for.

    Returns:
        int: The index of the matching insight, or -1 if not found.
    """
    for i, insight in enumerate(insights):
        if insight["insight"] == operation_rule_text:
            return i
    return -1


def remove_err_operations(
    insights: List[Dict[str, Any]], operations: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Cleans a list of rule operations by removing or modifying erroneous entries.

    This function iterates through a list of operations intended to modify a set of insights. It removes operations that are incorrect or not applicable (e.g., attempting to add a rule that already exists) and modifies certain operations based on their context (e.g., changing an EDIT to AGREE if the edited rule matches an existing rule). The goal is to ensure that the resulting list of operations is coherent and can be applied to update the insights without causing inconsistencies.

    Parameters:
        insights (List[Dict[str, Any]]): A list of tuples representing the existing insights. Each tuple contains the rule text and an associated numeric value, which could represent the rule's strength or priority.
        operations (List[Tuple[str, str]]): A list of tuples representing the operations to be performed on the insights. Each tuple contains an operation type (ADD, REMOVE, EDIT, AGREE) and the associated rule text or modification.

    Returns:
        List[Tuple[str, str]]: A cleaned list of operations where erroneous or inapplicable operations have been removed or modified to ensure consistency and correctness when applied to the set of existing insights.
    """
    corrected_operations = []
    for operation, text in operations.copy():
        operation_type = operation.split(" ")[0]
        insight_idx = int(operation.split(" ")[1]) if " " in operation else None
        index = retrieve_insight_index(insights, text)

        # ADDing an insight that doesn't exist.
        if operation_type == "ADD" and retrieve_insight_index(insights, text) == -1:
            corrected_operations.append((operation, text))
        # REMOVEing or AGREEing with an insight given that it exists.
        elif (operation_type == "REMOVE" or operation_type == "AGREE") and index != -1:
            corrected_operations.append((operation, text))
        # EDITing an insight (AGREEing) given that it exists.
        elif operation_type == "EDIT" and index != -1:
            corrected_operations.append((f"AGREE {index}", text))
        # EDITing an insight given:
        # - it doesn't exist (text match) in the insights
        # - the insight index to EDIT is not None
        # - the insight index to EDIT is less than or equal to the length of insights (within range of the length of the insights)
        elif (
            operation_type == "EDIT"
            and insight_idx is not None
            and insight_idx <= len(insights)
        ):
            corrected_operations.append((operation, text))

    return corrected_operations


def log_llm_io(
    response,
    context: str = "",
    verbose: bool = False,
    truncate_length: Optional[int] = None,
):
    """
    Log LLM input/output with rich formatting.

    Args:
        response: LLM response object with input_text and output_text
        context: Context string for the log
        verbose: Whether to actually log (if False, returns early)
        truncate_length: Maximum length for input/output text before truncating
    """
    if not verbose:
        return
    input_text = str(response.input_text)
    output_text = response.output_text
    if truncate_length is not None:
        if len(input_text) > truncate_length:
            input_text = input_text[:truncate_length] + "..."
        if len(output_text) > truncate_length:
            output_text = output_text[:truncate_length] + "..."
    # Escape context for rich markup
    context_escaped = escape(context)
    content = f"[bold blue]LLM {context_escaped}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{output_text}"
    console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))
