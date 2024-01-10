"""Functional module for Reflexion."""
import re

from typing import List, Optional, Tuple

import tiktoken

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from tiktoken.core import Encoding

from discussion_agents.cog.prompts.reflexion import (
    COT_AGENT_REFLECT_INSTRUCTION,
    COT_REFLECT_INSTRUCTION,
    LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
)

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


def _format_step(step: str) -> str:
    """Formats a step string by stripping leading/trailing newlines and spaces, and replacing internal newlines with empty space.

    Args:
        step (str): The step string to be formatted.

    Returns:
        str: The formatted step string.
    """
    # Remove leading/trailing newlines and spaces, and replace internal newlines with empty space.
    return step.strip("\n").strip().replace("\n", "")


def _parse_action(string: str) -> Optional[Tuple[str, str]]:
    """Parses an action string into an action type and its argument.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Optional[Tuple[str, str]]: A tuple containing the action type and argument, or None if parsing fails.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        print("uh oh")
        return None


def _prompt_cot_agent(
    llm: BaseChatModel,
    examples: str,
    reflections: str,
    context: str,
    question: str,
    scratchpad: str,
) -> str:
    """Generates a CoT prompt for thought and action.

    Args:
        llm (BaseChatModel): The language model to be used for generating the reflection.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        context (str): The context of the conversation or query.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.

    Returns:
        str: The generated reflection prompt.
    """
    prompt = PromptTemplate(
        input_variables=[
            "examples",
            "reflections",
            "context",
            "question",
            "scratchpad",
        ],
        template=COT_AGENT_REFLECT_INSTRUCTION,
    ).format(
        examples=examples,
        reflections=reflections,
        context=context,
        question=question,
        scratchpad=scratchpad,
    )

    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    return _format_step(out)


def _prompt_cot_reflection(
    llm: BaseChatModel, examples: str, context: str, question: str, scratchpad: str
) -> str:
    """Generates a reflection prompt.

    Args:
        llm (BaseChatModel): The language model to be used for generating the reflection.
        examples (str): Example inputs for the prompt template.
        context (str): The context of the conversation or query.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.

    Returns:
        str: The generated reflection prompt.
    """
    prompt = PromptTemplate(
        input_variables=["examples", "context", "question", "scratchpad"],
        template=COT_REFLECT_INSTRUCTION,
    ).format(
        examples=examples, context=context, question=question, scratchpad=scratchpad
    )

    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    return _format_step(out)


def reflect_last_attempt(scratchpad: str) -> List[str]:
    """Performs a reflection based on the last attempt (scratchpad).

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content from the last attempt.

    Returns:
        List[str]: A list with the scratchpad content.
    """
    return [scratchpad]


def reflect_reflexion(
    llm: BaseChatModel,
    reflections: List[str],
    examples: str,
    context: str,
    question: str,
    scratchpad: str,
) -> List[str]:
    """Perform reflexion-based reflecting.

    This function uses a language model to generate a new reflection based on the provided context, question,
    and scratchpad. The new reflection is added to the existing list of reflections.

    Args:
        llm (BaseChatModel): The language model used for generating the reflection.
        reflections (List[str]): Existing list of reflections.
        examples (str): Example inputs for the prompt template.
        context (str): The context of the conversation or query.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.

    Returns:
        List[str]: An updated list of reflections.
    """
    new_reflection = _prompt_cot_reflection(
        llm=llm,
        examples=examples,
        context=context,
        question=question,
        scratchpad=scratchpad,
    )
    reflections += [new_reflection]
    return reflections


def reflect_last_attempt_and_reflexion(
    llm: BaseChatModel,
    examples: str,
    context: str,
    question: str,
    scratchpad: str,
) -> List[str]:
    """Performs reflection with the reflection of the last attempt and reflexion.

    Args:
        llm (BaseChatModel): The language model used for generating the new reflection.
        examples (str): Example inputs for the prompt template.
        context (str): The context of the conversation or query.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.

    Returns:
        List[str]: A list with the new reflections.
    """
    reflections = [
        _prompt_cot_reflection(
            llm=llm,
            examples=examples,
            context=context,
            question=question,
            scratchpad=scratchpad,
        )
    ]
    return reflections


def reflect(
    strategy: str,
    llm: BaseChatModel,
    reflections: List[str],
    examples: str,
    context: str,
    question: str,
    scratchpad: str,
) -> List[str]:
    """Performs reflection based on a specified strategy using provided context, question, and scratchpad.

    This function orchestrates different types of reflections based on the strategy provided. It either reflects on the
    last attempt, generates a new reflexion, or combines both approaches. Depending on the strategy, it either uses
    the existing reflections, modifies them, or generates new reflections using the provided language model.

    Args:
        strategy (str): The reflection strategy to be used ('last_attempt', 'reflexion', or 'last_attempt_and_reflexion').
        llm (BaseChatModel): The language model used for generating new reflections.
        reflections (List[str]): A list of existing reflections.
        examples (str): Example inputs for the prompt template.
        context (str): The context of the conversation or query.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.

    Returns:
        List[str]: A tuple containing the updated list of reflections.

    Raises:
        NotImplementedError: If an unknown reflection strategy is specified.

    Strategy-Specific Parameters:
        - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
        - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
        - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
          It first formats the last attempt using 'question' and 'scratchpad', then adds a new reflexion using all the parameters.
    """
    if strategy == "last_attempt":
        reflections = reflect_last_attempt(scratchpad)
    elif strategy == "reflexion":
        reflections = reflect_reflexion(
            llm=llm,
            reflections=reflections,
            examples=examples,
            context=context,
            question=question,
            scratchpad=scratchpad,
        )
    elif strategy == "last_attempt_and_reflexion":
        reflections = reflect_last_attempt_and_reflexion(
            llm=llm,
            examples=examples,
            context=context,
            question=question,
            scratchpad=scratchpad,
        )
    else:
        raise NotImplementedError(f"Unknown reflection strategy: {strategy}.")

    return reflections
