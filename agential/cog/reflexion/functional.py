"""Functional module for Reflexion."""

import re

from typing import Any, Dict, List, Tuple, Union

import tiktoken

from tiktoken.core import Encoding

from agential.cog.reflexion.output import (
    ReflexionCoTStepOutput,
    ReflexionReActStepOutput,
)
from agential.cog.reflexion.prompts import (
    LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
)
from agential.llm.llm import BaseLLM, Response
from agential.utils.parse import remove_newline

gpt3_5_turbo_enc = tiktoken.encoding_for_model(
    "gpt-3.5-turbo"
)  # https://openai.com/blog/gpt-4-api-general-availability


def _truncate_scratchpad(
    scratchpad: str, n_tokens: int = 1600, tokenizer: Encoding = gpt3_5_turbo_enc
) -> str:
    """Truncates the scratchpad content to fit within a specified token limit.

    This function splits the scratchpad content into lines, filters out lines starting with 'Observation',
    and sorts them by token count. It then truncates the observations if the total token count exceeds the limit.

    Args:
        scratchpad (str): The scratchpad content to be truncated.
        n_tokens (int, optional): The maximum number of tokens allowed. Defaults to 1600.
        tokenizer (Encoding, optional): The tiktoken tokenizer used for counting tokens. Defaults to tiktoken's "gpt-3.5-turbo".

    Returns:
        str: The truncated scratchpad content.
    """
    # Split the scratchpad content into lines.
    lines = scratchpad.split("\n")
    # Filter out lines starting with 'Observation'.
    observations = filter(lambda x: x.startswith("Observation"), lines)
    # Sort observations by token count.
    observations_by_tokens = sorted(
        observations, key=lambda x: len(tokenizer.encode(x))
    )
    # Truncate observations if total token count exceeds limit.
    while len(tokenizer.encode("\n".join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        # Replace the largest observation with a truncated message.
        lines[ind] = (
            largest_observation.split(":")[0] + ": [truncated wikipedia excerpt]"
        )
    return "\n".join(lines)


def _format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    """Formats a list of reflection strings into a single formatted string.

    Args:
        reflections (List[str]): A list of reflection strings to be formatted.
        header (str, optional): A header to prepend to the formatted reflections. Defaults to REFLECTION_HEADER.

    Returns:
        str: The formatted string of reflections.
    """
    # Return formatted reflections if not empty.
    if reflections:
        return (
            header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])
        )
    else:
        return ""


def _format_last_attempt(
    question: str,
    scratchpad: str,
    header: str = LAST_TRIAL_HEADER,
    tokenizer: Encoding = gpt3_5_turbo_enc,
) -> str:
    """Formats the last attempt using the provided question and scratchpad content.

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content of the last attempt.
        header (str, optional): A header to prepend to the formatted last attempt. Defaults to LAST_TRIAL_HEADER.
        tokenizer (Encoding, optional): The tokenizer used for processing the scratchpad. Defaults to gpt3_5_turbo_enc.

    Returns:
        str: The formatted last attempt.
    """
    # Format the last attempt using the provided question and scratchpad.
    return (
        header
        + f"Question: {question}\n"
        + _truncate_scratchpad(scratchpad, tokenizer=tokenizer).strip("\n").strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


def _build_cot_agent_prompt(
    examples: str,
    reflections: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionCoT prompt template for the agent.

    This function formats a predefined prompt template (REFLEXION_COT_INSTRUCTION or
    REFLEXION_COT_INSTRUCTION_NO_CONTEXT) with examples,
    the provided question, and a scratchpad.

    Args:
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        examples=examples,
        reflections=reflections,
        question=question,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_cot_agent(
    llm: BaseLLM,
    examples: str,
    reflections: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a CoT prompt for thought and action.

    Used with ReflexionCoT.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_cot_agent_prompt(
        examples=examples,
        reflections=reflections,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def _build_cot_reflection_prompt(
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionCoT prompt template for reflection.

    Args:
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_cot_reflection(
    llm: BaseLLM,
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a reflection prompt.

    Used with ReflexionCoT.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_cot_reflection_prompt(
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def cot_reflect_last_attempt(scratchpad: str) -> Tuple[List[str], None]:
    """Performs a reflection based on the last attempt (scratchpad).

    Used with ReflexionCoT.

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content from the last attempt.

    Returns:
        Tuple[List[str], None]: A list with the scratchpad content.
    """
    return [scratchpad], None


def cot_reflect_reflexion(
    llm: BaseLLM,
    reflections: List[str],
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Tuple[List[str], Response]:
    """Perform reflexion-based reflecting.

    Used with ReflexionCoT. This function uses a language model to generate a new reflection based on the provided context, question,
    and scratchpad. The new reflection is added to the existing list of reflections.

    Args:
        llm (BaseLLM): The language model used for generating the reflection.
        reflections (List[str]): Existing list of reflections.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}

    Returns:
        Tuple[List[str], Response]: An updated list of reflections and the Response.
    """
    new_reflection = _prompt_cot_reflection(
        llm=llm,
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )

    reflections += [remove_newline(new_reflection.choices[0].message.content)]
    return reflections, new_reflection


def cot_reflect_last_attempt_and_reflexion(
    llm: BaseLLM,
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Tuple[List[str], Response]:
    """Performs reflection with the reflection of the last attempt and reflexion.

    Used with ReflexionCoT.

    Args:
        llm (BaseLLM): The language model used for generating the new reflection.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        context (Optional[str]): The context of the conversation or query. Defaults to None.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}

    Returns:
        Tuple[List[str], Response]: An updated list of reflections and the Response.
    """
    new_reflection = _prompt_cot_reflection(
        llm=llm,
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )

    reflections = [remove_newline(new_reflection.choices[0].message.content)]
    return reflections, new_reflection


def _build_react_agent_prompt(
    question: str,
    examples: str,
    reflections: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionReAct prompt template for the agent.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
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
        reflections=reflections,
        scratchpad=scratchpad,
        max_steps=max_steps,
        **additional_keys,
    )

    return prompt


def _prompt_react_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    reflections: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a ReAct prompt for thought and action.

    Used with ReflexionReAct.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_react_agent_prompt(
        question=question,
        examples=examples,
        reflections=reflections,
        scratchpad=scratchpad,
        max_steps=max_steps,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def _is_halted(
    finished: bool,
    step_idx: int,
    question: str,
    scratchpad: str,
    examples: str,
    reflections: str,
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
        reflections (str): Reflections.
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
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=max_steps,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )
        )
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit


def _build_react_reflection_prompt(
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionReAct prompt template for reflection.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_react_reflection(
    llm: BaseLLM,
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a reflection prompt.

    Used with ReflexionReAct.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_react_reflection_prompt(
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out


def react_reflect_last_attempt(scratchpad: str) -> Tuple[List[str], None]:
    """Performs a reflection based on the last attempt (scratchpad).

    Used with ReflexionReAct.

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content from the last attempt.

    Returns:
        Tuple[List[str], None]: A Tuple with the scratchpad content and None.
    """
    return [scratchpad], None


def react_reflect_reflexion(
    llm: BaseLLM,
    reflections: List[str],
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Tuple[List[str], Response]:
    """Perform reflexion-based reflecting.

    Used with ReflexionReAct. This function uses a language model to generate a new reflection based on the provided context, question,
    and scratchpad. The new reflection is added to the existing list of reflections.

    Args:
        llm (BaseLLM): The language model used for generating the reflection.
        reflections (List[str]): Existing list of reflections.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Tuple[List[str], Response]: An updated tuple of reflections and model response.
    """
    new_reflection_out = _prompt_react_reflection(
        llm=llm,
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    new_reflection = remove_newline(new_reflection_out.choices[0].message.content)
    reflections += [new_reflection]
    return reflections, new_reflection_out


def react_reflect_last_attempt_and_reflexion(
    llm: BaseLLM,
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Tuple[List[str], Response]:
    """Performs reflection with the reflection of the last attempt and reflexion.

    Used with ReflexionReAct.

    Args:
        llm (BaseLLM): The language model used for generating the new reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Tuple[List[str], Response]: A list with the new reflections and model response.
    """
    new_reflection_out = _prompt_react_reflection(
        llm=llm,
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    reflections = [remove_newline(new_reflection_out.choices[0].message.content)]
    return reflections, new_reflection_out


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


def parse_math_code_action_cot(action: str) -> Tuple[str, str]:
    """Parses an action string to extract the action type and code content.

    Identifies action types (`Finish`) and extracts the
    corresponding code content enclosed within Markdown-style code blocks.
    The action type is case-insensitive and the code content is trimmed of
    leading and trailing whitespace.

    Args:
        action (str): The action string containing the action type and code content.

    Returns:
        Tuple[str, str]: A tuple containing the extracted action type (capitalized)
        and the extracted code content.
    """
    action_split = action.split("```python", maxsplit=1)
    match = re.search(r"\b(Finish)\b", action_split[0], re.IGNORECASE)

    action_type = match.group(0).lower().capitalize() if match else ""
    try:
        query = action_split[1].split("```")[0].strip() if action_type else ""
    except:
        action_type = ""
        query = ""

    return action_type, query


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


def accumulate_metrics_cot(steps: List[ReflexionCoTStepOutput]) -> Dict[str, Any]:
    """Accumulates metrics for ReflexionCoT.

    Args:
        steps (List[ReflexionCoTStepOutput]): List of ReflexionCoTStepOutput objects.

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
            step.thought_metrics.prompt_tokens
            + step.action_metrics.prompt_tokens
            + (step.reflection_metrics.prompt_tokens if step.reflection_metrics else 0)
        )
        total_completion_tokens += (
            step.thought_metrics.completion_tokens
            + step.action_metrics.completion_tokens
            + (
                step.reflection_metrics.completion_tokens
                if step.reflection_metrics
                else 0
            )
        )
        total_tokens += (
            step.thought_metrics.total_tokens
            + step.action_metrics.total_tokens
            + (step.reflection_metrics.total_tokens if step.reflection_metrics else 0)
        )
        total_prompt_cost += (
            step.thought_metrics.prompt_cost
            + step.action_metrics.prompt_cost
            + (step.reflection_metrics.prompt_cost if step.reflection_metrics else 0.0)
        )
        total_completion_cost += (
            step.thought_metrics.completion_cost
            + step.action_metrics.completion_cost
            + (
                step.reflection_metrics.completion_cost
                if step.reflection_metrics
                else 0.0
            )
        )
        total_cost += (
            step.thought_metrics.total_cost
            + step.action_metrics.total_cost
            + (step.reflection_metrics.total_cost if step.reflection_metrics else 0.0)
        )
        total_prompt_time += (
            step.thought_metrics.prompt_time
            + step.action_metrics.prompt_time
            + (step.reflection_metrics.prompt_time if step.reflection_metrics else 0.0)
        )

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }


def accumulate_metrics_react(
    steps: List[ReflexionReActStepOutput],
) -> Dict[str, Any]:
    """Accumulates metrics for ReflexionReAct.

    Args:
        steps (List[ReflexionReActStepOutput]): List of ReflexionReActStepOutput objects.

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
            sum([s.thought_metrics.prompt_tokens for s in step.steps])
            + sum([s.action_metrics.prompt_tokens for s in step.steps])
            + (step.reflection_metrics.prompt_tokens if step.reflection_metrics else 0)
        )
        total_completion_tokens += (
            sum([s.thought_metrics.completion_tokens for s in step.steps])
            + sum([s.action_metrics.completion_tokens for s in step.steps])
            + (
                step.reflection_metrics.completion_tokens
                if step.reflection_metrics
                else 0
            )
        )
        total_tokens += (
            sum([s.thought_metrics.total_tokens for s in step.steps])
            + sum([s.action_metrics.total_tokens for s in step.steps])
            + (step.reflection_metrics.total_tokens if step.reflection_metrics else 0)
        )
        total_prompt_cost += (
            sum([s.thought_metrics.prompt_cost for s in step.steps])
            + sum([s.action_metrics.prompt_cost for s in step.steps])
            + (step.reflection_metrics.prompt_cost if step.reflection_metrics else 0.0)
        )
        total_completion_cost += (
            sum([s.thought_metrics.completion_cost for s in step.steps])
            + sum([s.action_metrics.completion_cost for s in step.steps])
            + (
                step.reflection_metrics.completion_cost
                if step.reflection_metrics
                else 0.0
            )
        )
        total_cost += (
            sum([s.thought_metrics.total_cost for s in step.steps])
            + sum([s.action_metrics.total_cost for s in step.steps])
            + (step.reflection_metrics.total_cost if step.reflection_metrics else 0.0)
        )
        total_prompt_time += (
            sum([s.thought_metrics.prompt_time for s in step.steps])
            + sum([s.action_metrics.prompt_time for s in step.steps])
            + (step.reflection_metrics.prompt_time if step.reflection_metrics else 0.0)
        )

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }
