"""Functional module for CRITIC."""

from typing import Any, Dict, List

from agential.agent.critic.output import CriticStepOutput
from agential.llm.llm import BaseLLM, Response


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
) -> Response:
    """Prompts the agent to answer a question using the language model.

    Parameters:
        llm (BaseLLM): The language model to use for generating the answer.
        question (str): The question to be answered.
        examples (str): Contextual examples relevant to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The answer from the language model, with no leading or trailing whitespace.
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
) -> Response:
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
        Response: The critique from the language model, with no leading or trailing whitespace.
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


def accumulate_metrics(steps: List[CriticStepOutput]) -> Dict[str, Any]:
    """Accumulates various metrics from a set of responses and experiences.

    This function takes in lists of comparison responses, success responses, and experiences, and calculates various metrics such as total prompt tokens, completion tokens, total tokens, prompt cost, completion cost, total cost, and prompt time. The results are returned as a dictionary.

    Parameters:
        steps (List[CriticStepOutput]): A list of CriticStepOutput objects containing the comparison responses, success responses, and experiences.

    Returns:
        Dict[str, Any]: A dictionary containing the accumulated metrics.
    """
    total_prompt_tokens = 0.0
    total_completion_tokens = 0.0
    total_tokens = 0.0
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_cost = 0.0
    total_prompt_time = 0.0

    for step in steps:
        total_prompt_tokens += sum(
            [answer.prompt_tokens for answer in step.answer_response]
        ) + sum([answer.prompt_tokens for answer in step.critique_response])
        total_completion_tokens += sum(
            [answer.completion_tokens for answer in step.answer_response]
        ) + sum([answer.completion_tokens for answer in step.critique_response])
        total_tokens += sum(
            [answer.total_tokens for answer in step.answer_response]
        ) + sum([answer.total_tokens for answer in step.critique_response])
        total_prompt_cost += sum(
            [answer.prompt_cost for answer in step.answer_response]
        ) + sum([answer.prompt_cost for answer in step.critique_response])
        total_completion_cost += sum(
            [answer.completion_cost for answer in step.answer_response]
        ) + sum([answer.completion_cost for answer in step.critique_response])
        total_cost += sum([answer.total_cost for answer in step.answer_response]) + sum(
            [answer.total_cost for answer in step.critique_response]
        )
        total_prompt_time += sum(
            [answer.prompt_time for answer in step.answer_response]
        ) + sum([answer.prompt_time for answer in step.critique_response])

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_prompt_cost": total_prompt_cost,
        "total_completion_cost": total_completion_cost,
        "total_cost": total_cost,
        "total_prompt_time": total_prompt_time,
    }
