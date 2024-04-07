"""Functional module for CRITIC."""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from discussion_agents.cog.prompts.critic import (
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_FORMAT_HOTPOTQA
)


def _build_agent_prompt(
    question: str,
    examples: str, 
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA
) -> str:
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    prompt: str = CRITIC_INSTRUCTION_HOTPOTQA
) -> str:
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        prompt=prompt
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content

    return out


def _build_critique_prompt(
    question: str,
    examples: str, 
    answer: str,
    prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA
) -> str:
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        answer=answer
    )
    return prompt


def _prompt_critique(
    llm: BaseChatModel,
    question: str,
    examples: str,
    answer: str,
    prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA
) -> str:
    prompt = _build_critique_prompt(
        question=question,
        examples=examples,
        answer=answer,
        prompt=prompt
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content

    return out


def _build_critique_format_prompt(
    question: str,
    examples: str, 
    answer: str,
    critique: str,
    prompt: str = CRITIC_CRITIQUE_FORMAT_HOTPOTQA
) -> str:
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique
    )
    return prompt
