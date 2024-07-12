"""Functional module for Self-Refine."""

from typing import Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.prompt import PromptTemplate


def _build_agent_prompt(
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a formatted prompt for the agent based on the question and provided fewshot examples.

    Parameters:
        question (str): The main question for which the agent is to generate an answer.
        examples (str): Pre-formatted few-shot examples that provide context for the question.
        prompt (str): The base template string into which all other components will be inserted.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The fully constructed and formatted prompt ready to be processed by the agent.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        **additional_keys,
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Generates a response from the LLM based on a given question with fewshot examples.

    This function creates a prompt using `_build_agent_prompt` and then gets the LLM's
    output.

    Args:
        llm (BaseChatModel): The language model to be prompted.
        question (str): The main question for which the agent is to generate an answer.
        examples (str): Pre-formatted few-shot examples that provide context for the question.
        prompt (str): The base template string into which all other components will be inserted.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The processed response from the language model.
    """
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print("<PROMPT AGENT====================================================>")
    print(prompt)
    print("<PROMPT AGENT====================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print("<OUT AGENT====================================================>")
    print(repr(out))
    print("<OUT AGENT====================================================>")
    assert isinstance(out, str)
    return out.strip()


def _build_critique_prompt(
    question: str,
    examples: str,
    answer: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Builds critique prompt.

    This function compiles a detailed prompt with contextual examples and a specific question format, then
    prompts the language model for a response.

    Parameters:
        llm (BaseChatModel): The language model to prompt for a response.
        question (str): The question to be answered by the language model.
        examples (str): Pre-formatted examples that provide context to the question.
        answer (str): The answer to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The language model's response to the question, trimmed of extraneous whitespace.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        answer=answer,
        **additional_keys,
    )
    return prompt


def _prompt_critique(
    llm: BaseChatModel,
    question: str,
    examples: str,
    answer: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Requests critique from the language model based on a provided answer and contextual examples.

    A critique prompt is constructed using the provided examples and answer.

    Parameters:
        llm (BaseChatModel): The language model to prompt for critique.
        question (str): The question to be answered by the language model.
        examples (str): Contextual examples related to the answer.
        answer (str): The answer for which critique is being sought.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The language model's critique, with no leading or trailing whitespace.
    """
    prompt = _build_critique_prompt(
        question=question,
        examples=examples,
        answer=answer,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out.strip()


def _build_refine_prompt(
    question: str,
    examples: str,
    answer: str,
    critique: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Builds a refinement prompt.

    Parameters:
        llm (BaseChatModel): The language model to prompt for a response.
        question (str): The question to be answered by the language model.
        examples (str): Pre-formatted examples that provide context to the question.
        critique (str): The critique on the answer.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The language model's response to the question, trimmed of extraneous whitespace.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        **additional_keys,
    )
    return prompt


def _prompt_refine(
    llm: BaseChatModel,
    question: str,
    examples: str,
    answer: str,
    critique: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Refines answer based on critique from the language model.

    A refine prompt is constructed using the provided answer, examples, and critique.

    Parameters:
        llm (BaseChatModel): The language model to prompt for critique.
        question (str): The question to be answered by the language model.
        examples (str): Contextual examples related to the answer.
        answer (str): The answer for which critique is being sought.
        critique (str): The critique on the answer.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The language model's critique, with no leading or trailing whitespace.
    """
    prompt = _build_refine_prompt(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print("<PROMPT REFINE====================================================>")
    print(prompt)
    print("<PROMPT REFINE====================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print("<OUT REFINE====================================================>")
    print(repr(out))
    print("<OUT REFINE====================================================>")
    assert isinstance(out, str)
    return out.strip()
