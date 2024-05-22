"""Functional module for CRITIC."""

import builtins
import sys

from typing import Any, Dict, List, Optional, Tuple

import func_timeout

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage

from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
)


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


# Ref: https://github.com/microsoft/ProphetNet/blob/master/CRITIC/src/tools/interpreter_api.py.
def safe_execute(
    code_string: str,
    keys: Optional[List[str]] = None,
) -> Tuple[List[Any], str]:
    """Executes the provided Python code string in a safe manner with a timeout and returns specified variables from the execution.

    Args:
        code_string (str): Python code to execute.
        keys (Optional[List[str]]): A list of variable names whose values are to be returned after execution. If None, the function tries to return a variable named 'answer'.

    Returns:
        tuple: A tuple containing the result(s) of the specified variable(s) and a status message. If an exception occurs or timeout happens, it returns None for the result.
    """
    safe_globals: Dict[str, Any] = {"__builtins__": builtins, "sys": sys}

    def execute(x: str) -> Tuple[Optional[Any], str]:
        """Executes the code string with python exec()."""
        try:
            exec(x, safe_globals)
            if keys is None:
                an = [safe_globals.get("answer", None)]
            else:
                an = [safe_globals.get(k, None) for k in keys]
            return an, "Done"
        except BaseException as e:
            return [None], repr(e)

    try:
        an, report = func_timeout.func_timeout(3, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        an = [None]
        report = "TimeoutError: execution timeout"

    return an, report


def _build_agent_prompt(
    question: str,
    examples: str,
    additional_keys: Dict[str, str] = {},
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
) -> str:
    """Builds a prompt for questioning the agent using a template.

    Parameters:
        question (str): The question to be answered by the agent.
        examples (str): Contextual examples related to the question.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
        prompt (str): Prompt template string. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.

    Returns:
        str: A formatted prompt ready for use with the language model.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question, examples=examples, **additional_keys
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    additional_keys: Dict[str, str] = {},
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
) -> str:
    """Prompts the agent to answer a question using the language model.

    Parameters:
        llm (BaseChatModel): The language model to use for generating the answer.
        question (str): The question to be answered.
        examples (str): Contextual examples relevant to the question.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
        prompt (str): Prompt template string. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.

    Returns:
        str: The answer from the language model, with no leading or trailing whitespace.
    """
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        additional_keys=additional_keys,
        prompt=prompt,
    )
    print("<PROMPT AGENT===========================================================>")
    print(prompt)
    print("<PROMPT AGENT===========================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print("<OUT AGENT===========================================================>")
    print(repr(out))
    print("<OUT AGENT===========================================================>")
    assert isinstance(out, str)
    return out


def _build_critique_prompt(
    question: str,
    examples: str,
    answer: str,
    critique: str = "",
    additional_keys: Dict[str, str] = {},
    prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
) -> str:
    """Builds a critique prompt for the agent using a template.

    Parameters:
        question (str): The original question related to the answer.
        examples (str): Contextual examples used in the question.
        answer (str): The agent's answer to the question.
        critique (str, optional): Additional critique information.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
        prompt (str): Prompt template string. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.

    Returns:
        str: A formatted critique prompt ready for use with the language model.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        **additional_keys,
    )
    return prompt


def _prompt_critique(
    llm: BaseChatModel,
    question: str,
    examples: str,
    answer: str,
    critique: str = "",
    additional_keys: Dict[str, str] = {},
    prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
) -> str:
    """Prompts the agent for a critique of an answer using the language model.

    Parameters:
        llm (BaseChatModel): The language model to use for generating the critique.
        question (str): The question related to the answer.
        examples (str): Contextual examples related to the question.
        answer (str): The answer to critique.
        critique (str, optional): Initial critique to refine the response.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
        prompt (str): Prompt template string. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.

    Returns:
        str: The critique from the language model, with no leading or trailing whitespace.
    """
    prompt = _build_critique_prompt(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        additional_keys=additional_keys,
        prompt=prompt,
    )
    print("<PROMPT CRITIC===========================================================>")
    print(prompt)
    print("<PROMPT CRITIC===========================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print("<OUT CRITIC===========================================================>")
    print(repr(out))
    print("<OUT CRITIC===========================================================>")
    assert isinstance(out, str)
    return out
