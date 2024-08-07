"""Functional module for CRITIC."""

from typing import Dict

from agential.llm.llm import BaseLLM


# Ref: https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/program/utils.py.
def remove_comment(code: str) -> str:
    """Removes all comment lines and empty lines from the given block of code.

    Args:
        code (str): A string containing the block of code from which comments and empty lines will be removed.

    Returns:
        str: The code with all comment lines that start with '#' and empty lines removed.
    """
    code_lines = code.split("\n")
    code_lines = [line for line in code_lines if not line.startswith("#")]
    code_lines = [line for line in code_lines if line.strip() != ""]
    return "\n".join(code_lines)


def _build_agent_prompt(
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Builds a prompt for questioning the agent using a template.

    Parameters:
        question (str): The question to be answered by the agent.
        examples (str): Contextual examples related to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt ready for use with the language model.
    """
    prompt = prompt.format(question=question, examples=examples, **additional_keys)
    return prompt


def _prompt_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Prompts the agent to answer a question using the language model.

    Parameters:
        llm (BaseLLM): The language model to use for generating the answer.
        question (str): The question to be answered.
        examples (str): Contextual examples relevant to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The answer from the language model, with no leading or trailing whitespace.
    """
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)

    return out


def _build_critique_prompt(
    question: str,
    examples: str,
    answer: str,
    critique: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Builds a critique prompt for the agent using a template.

    Parameters:
        question (str): The original question related to the answer.
        examples (str): Contextual examples used in the question.
        answer (str): The agent's answer to the question.
        critique (str, optional): Additional critique information.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted critique prompt ready for use with the language model.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        **additional_keys,
    )
    return prompt


def _prompt_critique(
    llm: BaseLLM,
    question: str,
    examples: str,
    answer: str,
    critique: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Prompts the agent for a critique of an answer using the language model.

    Parameters:
        llm (BaseLLM): The language model to use for generating the critique.
        question (str): The question related to the answer.
        examples (str): Contextual examples related to the question.
        answer (str): The answer to critique.
        critique (str, optional): Initial critique to refine the response.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The critique from the language model, with no leading or trailing whitespace.
    """
    prompt = _build_critique_prompt(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)

    return out
