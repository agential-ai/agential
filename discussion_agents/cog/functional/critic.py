"""Functional module for CRITIC."""

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from discussion_agents.cog.prompts.critic import (
    CRITIC_INSTRUCTION_HOTPOTQA,
    CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    CRITIC_INSTRUCTION_TRIVIAQA,
    CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
    
)

BENCHMARK_PROMPTS = {
    'hotpotqa': CRITIC_INSTRUCTION_HOTPOTQA,
    'triviaqa': CRITIC_INSTRUCTION_TRIVIAQA,
    # Add more mappings as necessary
}

BENCHMARK_PROMPTS_CRITIQUE ={
    'hotpotqa': CRITIC_CRITIQUE_INSTRUCTION_HOTPOTQA,
    'triviaqa': CRITIC_CRITIQUE_INSTRUCTION_TRIVIAQA,
}



def _build_agent_prompt(
    question: str,
    examples: str, 
    benchmark: str
) -> str:
    if benchmark in BENCHMARK_PROMPTS:
        prompt = BENCHMARK_PROMPTS[benchmark]
        
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    formatted_prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples
    )
    return formatted_prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    benchmark: str
) -> str:

    formatted_prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        benchmark=benchmark
    )

    out = llm(
        [
            HumanMessage(
                content=formatted_prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out


def _build_critique_prompt(
    question: str,
    examples: str, 
    answer: str,
    benchmark: str,
    critique: str = ""
) -> str:
    
    if benchmark in BENCHMARK_PROMPTS_CRITIQUE:
        prompt = BENCHMARK_PROMPTS_CRITIQUE[benchmark]
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    # print('_build_critique_prompt')
    # print('-------------------------')
    # print('question',question)
    # print('-------------------------')
    # print('example',examples)
    # print('-------------------------')
    # print('answer',answer)
    # print('--------------------')
    # print('benchmark :',prompt)
    # print('--------------------')
    # print('critique :',critique)
    
    formatted_prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique
    )
    return formatted_prompt


def _prompt_critique(
    llm: BaseChatModel,
    question: str,
    examples: str,
    answer: str,
    benchmark: str,
    critique: str = ""
) -> str:
    
    # print('question',question)
    # print('-------------------------')
    # print('example',examples)
    # print('-------------------------')
    # print('answer',answer)
    # print('--------------------')
    # print('benchmark',benchmark)
    # print('--------------------')
    # print('critique',critique)
    
    formatted_prompt = _build_critique_prompt(
        question=question,
        examples=examples,
        answer=answer,
        critique=critique,
        benchmark=benchmark
    )

    # print('formatted_prompt',formatted_prompt)

    out = llm(
        [
            HumanMessage(
                content=formatted_prompt,
            )
        ]
    ).content

    assert isinstance(out, str)

    return out



