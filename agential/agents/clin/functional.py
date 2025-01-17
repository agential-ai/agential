"""CLIN functional module."""

import re

from typing import Any, Dict, List, Optional, Tuple

from agential.agents.clin.output import CLINStepOutput
from agential.core.llm import BaseLLM, Response


def _build_react_agent_prompt(
    question: str,
    examples: str,
    summaries: str,
    scratchpad: str,
    max_steps: int,
    summary_system: str,
    meta_summaries: str,
    meta_summary_system: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a CLIN prompt template for the agent.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        summaries (str): Summaries of previous steps.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        summary_system (str): System prompt for summarization.
        meta_summaries (str): Summaries of previous steps.
        meta_summary_system (str): System prompt for meta-summarization.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        summaries=summaries,
        scratchpad=scratchpad,
        max_steps=max_steps,
        summary_system=summary_system,
        meta_summaries=meta_summaries,
        meta_summary_system=meta_summary_system,
        **additional_keys,
    )

    return prompt


def _prompt_react_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    summaries: str,
    scratchpad: str,
    max_steps: int,
    summary_system: str,
    meta_summaries: str,
    meta_summary_system: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a ReAct prompt for thought and action.

    Used with CLIN.

    Args:
        llm (BaseLLM): The language model to be used for generation.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        summaries (str): Summaries of previous steps.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        summary_system (str): System prompt for summarization.
        meta_summaries (str): Summaries of previous steps.
        meta_summary_system (str): System prompt for meta-summarization.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated prompt.
    """
    prompt = _build_react_agent_prompt(
        question=question,
        examples=examples,
        summaries=summaries,
        scratchpad=scratchpad,
        max_steps=max_steps,
        summary_system=summary_system,
        meta_summaries=meta_summaries,
        meta_summary_system=meta_summary_system,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)

    return out


def _build_summary_prompt(
    question: str,
    previous_trials: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a CLIN prompt template for the agent.

    Args:
        question (str): The question being addressed.
        previous_trials (str): The scratchpad content related to the question.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        previous_trials=previous_trials,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_summary(
    llm: BaseLLM,
    question: str,
    previous_trials: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Summarizes the scratchpad content.

    Used with CLIN.

    Args:
        llm (BaseLLM): The language model to be used for generation.
        question (str): The question being addressed.
        previous_trials (str): The scratchpad content related to the question.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated prompt.
    """
    prompt = _build_summary_prompt(
        question=question,
        previous_trials=previous_trials,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)

    return out


def _build_meta_summary_prompt(
    question: str,
    meta_summary_system: str,
    meta_summaries: str,
    previous_trials: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a CLIN prompt template for the agent.

    Args:
        question (str): The question being addressed.
        meta_summary_system (str): System prompt for summarization.
        meta_summaries (str): Summaries of previous steps.
        previous_trials (str): The scratchpad content related to the question.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        meta_summary_system=meta_summary_system,
        meta_summaries=meta_summaries,
        previous_trials=previous_trials,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_meta_summary(
    llm: BaseLLM,
    question: str,
    meta_summary_system: str,
    meta_summaries: str,
    previous_trials: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Summarizes the scratchpad content.

    Used with CLIN.

    Args:
        llm (BaseLLM): The language model to be used for generation.
        question (str): The question being addressed.
        meta_summary_system (str): System prompt for summarization.
        meta_summaries (str): Summaries of previous steps.
        previous_trials (str): The scratchpad content related to the question.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated prompt.
    """
    prompt = _build_meta_summary_prompt(
        question=question,
        meta_summary_system=meta_summary_system,
        meta_summaries=meta_summaries,
        previous_trials=previous_trials,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)

    return out


def _is_halted(
    finished: bool,
    step_idx: int,
    max_steps: int,
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on two conditions:
    completion (finished) or exceeding maximum steps.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_idx (int): Current step number.
        max_steps (int): Maximum allowed steps.

    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_idx > max_steps

    return finished or over_max_steps


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct and Reflexion.

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


def parse_math_code_action_react(
    action: str, action_types: List[str]
) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`, `Calculate`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.
        action_types (List[str]): List of action types to identify.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    pattern = r"\b(" + "|".join(action_types) + r")\b"
    match = re.search(pattern, action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


def accumulate_metrics(
    steps: List[CLINStepOutput],
    meta_summaries_response: Optional[Response],
) -> Dict[str, Any]:
    """Accumulates metrics for CLIN.

    Args:
        steps (List[CLINStepOutput]): List of CLINStepOutput objects.
        meta_summaries_response (Optional[Response]): Response from meta_summaries. Can be None.

    Returns:
        Dict[str, Any]: A dictionary containing the following accumulated metrics:
            - total_prompt_tokens (int): Total number of prompt tokens used.
            - total_completion_tokens (int): Total number of completion tokens generated.
            - total_tokens (int): Total number of tokens (prompt + completion).
            - total_prompt_cost (float): Total cost associated with prompts.
            - total_completion_cost (float): Total cost associated with completions.
            - total_cost (float): Total overall cost (prompt + completion).
            - total_prompt_time (float): Total time spent on prompts.
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_cost = 0.0
    total_prompt_time = 0.0

    for step in steps:
        total_prompt_tokens += (
            sum([s.thought_response.prompt_tokens for s in step.steps])
            + sum([s.action_response.prompt_tokens for s in step.steps])
            + step.summaries_response.prompt_tokens
        )
        total_completion_tokens += (
            sum([s.thought_response.completion_tokens for s in step.steps])
            + sum([s.action_response.completion_tokens for s in step.steps])
            + step.summaries_response.completion_tokens
        )
        total_tokens += (
            sum([s.thought_response.total_tokens for s in step.steps])
            + sum([s.action_response.total_tokens for s in step.steps])
            + step.summaries_response.total_tokens
        )
        total_prompt_cost += (
            sum([s.thought_response.prompt_cost for s in step.steps])
            + sum([s.action_response.prompt_cost for s in step.steps])
            + step.summaries_response.prompt_cost
        )
        total_completion_cost += (
            sum([s.thought_response.completion_cost for s in step.steps])
            + sum([s.action_response.completion_cost for s in step.steps])
            + step.summaries_response.completion_cost
        )
        total_cost += (
            sum([s.thought_response.total_cost for s in step.steps])
            + sum([s.action_response.total_cost for s in step.steps])
            + step.summaries_response.total_cost
        )
        total_prompt_time += (
            sum([s.thought_response.prompt_time for s in step.steps])
            + sum([s.action_response.prompt_time for s in step.steps])
            + step.summaries_response.prompt_time
        )

    if meta_summaries_response is not None:
        total_prompt_tokens += meta_summaries_response.prompt_tokens
        total_completion_tokens += meta_summaries_response.completion_tokens
        total_tokens += meta_summaries_response.total_tokens
        total_prompt_cost += meta_summaries_response.prompt_cost
        total_completion_cost += meta_summaries_response.completion_cost
        total_cost += meta_summaries_response.total_cost
        total_prompt_time += meta_summaries_response.prompt_time

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }
