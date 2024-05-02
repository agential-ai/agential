"""Functional module for CRITIC."""

from typing import Dict

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage

from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
)


def _prompt_agent(
    llm: BaseChatModel,
    keys: Dict[str, str],
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA,
) -> str:
    """Prompts the agent to answer a question using the language model.

    Parameters:
        llm (BaseChatModel): The language model to use for generating the answer.
        question (str): The question to be answered.
        examples (str): Contextual examples relevant to the question.
        prompt (str): Prompt template string. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.

    Returns:
        str: The answer from the language model, with no leading or trailing whitespace.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question, examples=examples
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out


def _prompt_critique(
    llm: BaseChatModel,
    question: str,
    examples: str,
    answer: str,
    critique: str = "",
    prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
) -> str:
    """Prompts the agent for a critique of an answer using the language model.

    Parameters:
        llm (BaseChatModel): The language model to use for generating the critique.
        question (str): The question related to the answer.
        examples (str): Contextual examples related to the question.
        answer (str): The answer to critique.
        critique (str, optional): Initial critique to refine the response.
        prompt (str): Prompt template string. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.

    Returns:
        str: The critique from the language model, with no leading or trailing whitespace.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question, examples=examples, answer=answer, critique=critique
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out
