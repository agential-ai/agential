import re

from typing import Any, Dict, List, Optional, Tuple

from tiktoken import Encoding

from agential.agents.clin.output import CLINReActStepOutput, CLINStepOutput
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

    print("prompt input _________________________________________________________")
    print(prompt)
    print("input _________________________________________________________")

    out = llm(prompt)

    print("prompt output _________________________________________________________")
    print(repr(out.output_text))
    print("output _________________________________________________________")

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

    print("prompt input _________________________________________________________")
    print(prompt)
    print("input _________________________________________________________")

    out = llm(prompt)

    print("prompt output _________________________________________________________")
    print(repr(out.output_text))
    print("output _________________________________________________________")

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

    print("prompt input _________________________________________________________")
    print(prompt)
    print("input _________________________________________________________")

    out = llm(prompt)

    print("prompt output _________________________________________________________")
    print(repr(out.output_text))
    print("output _________________________________________________________")

    return out


def _is_halted(
    finished: bool,
    step_idx: int,
    question: str,
    scratchpad: str,
    examples: str,
    summaries: str,
    summary_system: str,
    meta_summaries: str,
    meta_summary_system: str,
    max_steps: int,
    max_tokens: int,
    enc: Encoding,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on three conditions:
    completion (finished), exceeding maximum steps, or exceeding maximum token limit.
    The token limit is evaluated based on the encoded length of the prompt.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_idx (int): Current step number.
        question (str): The question being processed.
        scratchpad (str): The scratchpad content.
        examples (str): Fewshot examples.
        summaries (str): Summaries.
        summary_system (str): System prompt for summarization.
        meta_summaries (str): Summaries of previous steps.
        meta_summary_system (str): System prompt for summarization.
        max_steps (int): Maximum allowed steps.
        max_tokens (int): Maximum allowed token count.
        enc (Encoding): The encoder to calculate token length.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to

    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_idx > max_steps
    over_token_limit = (
        len(
            enc.encode(
                _build_react_agent_prompt(
                    examples=examples,
                    summaries=summaries,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=max_steps,
                    summary_system=summary_system,
                    meta_summaries=meta_summaries,
                    meta_summary_system=meta_summary_system,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )
        )
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit


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


def accumulate_metrics(
    steps: List[CLINStepOutput],
    meta_summaries_response: Optional[Response],
) -> Dict[str, Any]:
    """Accumulates metrics for CLIN.

    Args:
        steps (List[ClinStepOutput]): List of ClinStepOutput objects.
        meta_summaries_response (Optional[Response]): Response from meta_summaries.

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
