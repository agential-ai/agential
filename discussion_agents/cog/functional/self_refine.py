"""Functional module for Self-Refine."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain_core.messages.human import HumanMessage
from discussion_agents.cog.prompts.self_refine import (
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K
)

def _build_agent_prompt(
    question: str,
    examples: str,
    prompt: str = SELF_REFINE_INSTRUCTION_GSM8K,
) -> str:
    """Constructs a formatted prompt for the agent based on a template and provided components.

    The prompt is created by inserting specific content into placeholders in the template string. 
    This includes the current question to be answered, any preceding few-shot examples, and appropriate 
    separators and prefixes to structure the prompt for better understanding by the agent.

    Parameters:
        question (str): The main question for which the agent is to generate an answer.
        examples (str): Pre-formatted few-shot examples that provide context for the question.
        prompt (str): The base template string into which all other components will be inserted. This 
            template must have placeholders for the 'question', 'examples', 'question_prefix', 
            'intra_example_sep', and 'answer_prefix'. Defaults to SELF_REFINE_INSTRUCTION_GSM8K.

    Returns:
        str: The fully constructed and formatted prompt ready to be processed by the agent.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        examples=examples,
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    prompt: str = SELF_REFINE_INSTRUCTION_GSM8K,
) -> str:
    """Generates a response from the LLM based on a given question and scratchpad.

    This function creates a prompt using `_build_agent_prompt` and then gets the LLM's
    output. The newline characters in the output are removed before returning.

    Args:
        llm (BaseChatModel): The language model to be prompted.
        question (str): The main question for which the agent is to generate an answer.
        examples (str): Pre-formatted few-shot examples that provide context for the question.
        prompt (str): The base template string into which all other components will be inserted. This 
            template must have placeholders for the 'question', 'examples', 'question_prefix', 
            'intra_example_sep', and 'answer_prefix'. Defaults to SELF_REFINE_INSTRUCTION_GSM8K.

    Returns:
        str: The processed response from the language model.
    """
    prompt = _build_agent_prompt(
        question=question,
        examples=examples,
        prompt=prompt
    )
    print("<==============================================================>")
    print(prompt)
    print("<==============================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out.strip()


def _build_feedback_prompt(
    examples: str,
    solution: str,
    prompt: str = SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K
) -> str:
    """Invokes the language model to generate a response for the specified question using structured examples.

    This function compiles a detailed prompt with contextual examples and a specific question format, then
    prompts the language model for a response. It cleans up the response by stripping out any leading or
    trailing whitespace or newline characters.

    Parameters:
        llm (BaseChatModel): The language model to prompt for a response.
        question (str): The question to be answered by the language model.
        examples (str): Pre-formatted examples that provide context to the question.
        prompt (str): Prompt template string. Defaults to SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K.

    Returns:
        str: The language model's response to the question, trimmed of extraneous whitespace.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        examples=examples,
        solution=solution,
    )
    return prompt


def _prompt_feedback(
    llm: BaseChatModel,
    examples: str,
    solution: str,
    prompt: str = SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K
) -> str:
    """Requests feedback from the language model based on a provided solution and contextual examples.

    A feedback prompt is constructed using the provided solution, examples, and a feedback instruction.
    The language model is prompted with this structured request, and its output is cleaned of leading and
    trailing whitespace.

    Parameters:
        llm (BaseChatModel): The language model to prompt for feedback.
        examples (str): Contextual examples related to the solution.
        solution (str): The solution for which feedback is being sought.
        prompt (str): Prompt template string. Defaults to SELF_REFINE_FEEDBACK_INSTRUCTION_GSM8K.

    Returns:
        str: The language model's feedback, with no leading or trailing whitespace.
    """
    prompt = _build_feedback_prompt(
        examples=examples,
        solution=solution,
        prompt=prompt
    )
    print("<FEEDBACK==============================================================>")
    print(prompt)
    print("<FEEDBACK==============================================================>")
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return out.strip()