from typing import Dict
from agential.core.llm import BaseLLM, Response


def _build_agent_prompt(
    question: str,
    examples: str,
    summaries: str,
    meta_summaries: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a CLIN prompt template for the agent.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        summaries (str): Existing summaries.
        meta_summaries (str): Existing meta summaries.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        summaries=summaries,
        meta_summaries=meta_summaries,
        scratchpad=scratchpad,
        max_steps=max_steps,
        **additional_keys,
    )

    return prompt


def _prompt_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    summaries: str,
    meta_summaries: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a CLIN prompt for thought and action.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        summaries (str): Existing summaries.
        meta_summaries (str): Existing meta summaries.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        summaries=summaries,
        meta_summaries=meta_summaries,
        scratchpad=scratchpad,
        max_steps=max_steps,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def _build_summary_prompt(
    question: str,
    examples: str,
    summaries: str,
    meta_summaries: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a CLIN prompt template for the agent.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        summaries (str): Existing summaries.
        meta_summaries (str): Existing meta summaries.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        summaries=summaries,
        meta_summaries=meta_summaries,
        scratchpad=scratchpad,
        max_steps=max_steps,
        **additional_keys,
    )

    return prompt


def _prompt_summary(
    llm: BaseLLM,
    question: str,
    examples: str,
    summaries: str,
    meta_summaries: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a CLIN prompt for thought and action.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        summaries (str): Existing summaries.
        meta_summaries (str): Existing meta summaries.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_summary_prompt(
        question=question,
        examples=examples,
        summaries=summaries,
        meta_summaries=meta_summaries,
        scratchpad=scratchpad,
        max_steps=max_steps,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


