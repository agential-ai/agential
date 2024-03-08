"""Functional module for ReAct."""
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from tiktoken import Encoding
from typing import Optional

from discussion_agents.utils.parse import remove_newline


def _build_agent_prompt(question: str, scratchpad: str, examples: str , prompt_template: str) -> str:
    """Constructs a prompt template for the agent.

    This function formats a predefined prompt template (REACT_INSTRUCTION) with examples,
    the provided question, and a scratchpad.

    Args:
        question (str): The question to be included in the prompt.
        scratchpad (str): Additional scratchpad information to be included.
        examples (str): The example as a guide of how the test should be prompted.
        prompt_template (str): The template of the prompt that is inputted into scratchpad.
    Returns:
        str: A formatted prompt template ready for use.
    """

    prompt = PromptTemplate.from_template(prompt_template).format(
        examples=examples,
        question=question,
        scratchpad=scratchpad,
    )
    return prompt


def _prompt_agent(llm: BaseChatModel, question: str, scratchpad: str, examples: str, prompt_template: str) -> str:
    """Generates a response from the LLM based on a given question and scratchpad.

    This function creates a prompt using `_build_agent_prompt` and then gets the LLM's
    output. The newline characters in the output are removed before returning.

    Args:
        llm (BaseChatModel): The language model to be prompted.
        question (str): The question to ask the language model.
        scratchpad (str): Additional context or information for the language model.
        examples (str): The example used for specific benchmark for ai model to generate prompt accordingly.
        prompt_template (str): The template of the prompt that is inputted into scratchpad.
    Returns:
        str: The processed response from the language model.
    """
    prompt = _build_agent_prompt(question=question, scratchpad=scratchpad, examples=examples, prompt_template=prompt_template)
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ],
        stop = '\n'
    ).content
    assert isinstance(out, str)
    return out 


def _is_halted(
    finished: bool,
    step_n: int,
    max_steps: int,
    question: str,
    scratchpad: str,
    max_tokens: int,
    enc: Encoding,
    examples: str ,
    prompt_template: str
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on three conditions:
    completion (finished), exceeding maximum steps, or exceeding maximum token limit.
    The token limit is evaluated based on the encoded length of the prompt.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_n (int): Current step number.
        max_steps (int): Maximum allowed steps.
        question (str): The question being processed.
        scratchpad (str): The scratchpad content.
        max_tokens (int): Maximum allowed token count.
        enc (Encoding): The encoder to calculate token length.
        examples (str): The example of the output prompt.
        prompt_template (str): The template of the prompt.
    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_n > max_steps
    over_token_limit = (
        len(enc.encode(_build_agent_prompt(question=question, scratchpad=scratchpad, examples=examples, instruction=instruction)))
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit

def _process_ob(ob):
    """Processing string for better output prompt.
    Args:
        ob (str): The observation after the action.
    Returns:
        ob (str): The string goes into output
    """
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob




def _check_keyword(example: str = None):
    """checking the step utilized in example.
    Args:
        example (str): The example input of the generation function.
    Returns:
        step_occurence (list[int]): a list of number that indicate which step is utilised in the example
    """
    keyword = ['Thought', 'Action', 'Observation', 'Your task is to']
    example = example.split('\n')
    example = [line.strip() for line in example if line]
    step_occurrence = [0 , 0, 0, 0]
    i = 0
    while all(num < 2 for num in step_occurrence):
        test_case = example[i].split(':')[0]
        test_case = ''.join(char for char in test_case if not char.isdigit()).strip()
        if any(part.strip() in keyword for part in test_case) or test_case in keyword: 
            index = keyword.index(test_case)
            step_occurrence[index] += 1
        i += 1
    return step_occurrence

def prepare_action(action):
    action = action.replace('>', '').strip()
    if 'think' not in action:
        action = action.replace(' in ', ' in/on ')
    return action