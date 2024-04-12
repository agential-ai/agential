"""Functional module for CRITIC."""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage

from discussion_agents.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_HOTPOTQA,
)


def _build_agent_prompt(
    question: str, examples: str, prompt: str = CRITIC_INSTRUCTION_HOTPOTQA
) -> str:
    """Builds a prompt for questioning the agent using a template.

    Parameters:
        question (str): The question to be answered by the agent.
        examples (str): Contextual examples related to the question.
        prompt (str): Prompt template string. Defaults to CRITIC_INSTRUCTION_HOTPOTQA.

    Returns:
        str: A formatted prompt ready for use with the language model.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question, examples=examples
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
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
    prompt = _build_agent_prompt(question=question, examples=examples, prompt=prompt)
    print("PROMPT<===================================================================================>")
    print(prompt)
    print("PROMPT<===================================================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out


def _build_critique_prompt(
    question: str,
    examples: str,
    answer: str,
    critique: str = "",
    prompt: str = CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
) -> str:
    """Builds a critique prompt for the agent using a template.

    Parameters:
        question (str): The original question related to the answer.
        examples (str): Contextual examples used in the question.
        answer (str): The agent's answer to the question.
        critique (str, optional): Additional critique information.
        prompt (str): Prompt template string. Defaults to CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA.

    Returns:
        str: A formatted critique prompt ready for use with the language model.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question, examples=examples, answer=answer, critique=critique
    )
    return prompt


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
    prompt = _build_critique_prompt(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        prompt=prompt,
    )
    print("CRITIQUE PROMPT<===================================================================================>")
    print(prompt)
    print("CRITIQUE PROMPT<===================================================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content

    assert isinstance(out, str)

    return out
