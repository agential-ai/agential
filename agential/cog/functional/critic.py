"""Functional module for CRITIC."""

from typing import Dict

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage

from agential.utils.prompt import prompt_llm
from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
)


def _prompt_agent(
    llm: BaseChatModel,
    keys: Dict[str, str],
    prompt_template: str = CRITIC_INSTRUCTION_HOTPOTQA,
) -> str:
    """Prompts the agent to answer a question using the language model.

    Parameters:
        llm (BaseChatModel): The language model to use for generating the answer.
        keys (Dict[str, str]): The keys and values to format the prompt. Required keys are listed below.
        prompt_template (str): Prompt template string. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.

    Keys Required:
        - For QA Benchmarks:
            question (str): The question that the agent needs to answer.
            examples (str): Fewshot examples relevant to the question.

    Returns:
        str: The answer from the language model, with no leading or trailing whitespace.
    """
    return prompt_llm(
        llm=llm,
        keys=keys,
        prompt_template=prompt_template
    )


def _prompt_critique(
    llm: BaseChatModel,
    keys: Dict[str, str],
    prompt_template: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA
) -> str:
    """Prompts the agent for a critique of an answer using the language model.

    Parameters:
        llm (BaseChatModel): The language model to use for generating the critique.
        keys (Dict[str, str]): The keys and values to format the prompt. Required keys are listed below.
        prompt_template (str): Prompt template string. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.

    Keys Required:
        - For QA Benchmarks:
            question (str): The question related to the answer.
            examples (str): Fewshot examples related to the question.
            answer (str): The answer to critique.
            critique (str, optional): Critique to refine the response. Defaults to None.

    Returns:
        str: The critique from the language model.
    """
    return prompt_llm(
        llm=llm,
        keys=keys,
        prompt_template=prompt_template
    )