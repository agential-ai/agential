"""CoT functional module."""

from typing import Dict

from agential.llm.llm import BaseLLM, Response


def _build_prompt(
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Builds a prompt for questioning the llm using a template.

    Parameters:
        question (str): The question to be answered by the llm.
        examples (str): Contextual examples related to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt ready for use with the language model.
    """
    prompt = prompt.format(question=question, examples=examples, **additional_keys)
    return prompt


def _prompt_llm(
    llm: BaseLLM,
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Prompts the llm to answer a question using the language model.

    Parameters:
        llm (BaseLLM): The language model to use for generating the answer.
        question (str): The question to be answered.
        examples (str): Contextual examples relevant to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The answer from the language model, with no leading or trailing whitespace.
    """
    prompt = _build_prompt(
        question=question,
        examples=examples,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out
