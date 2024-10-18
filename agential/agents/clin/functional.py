from typing import Dict, Tuple

from tiktoken import Encoding
from agential.core.llm import BaseLLM, Response


def _build_react_agent_prompt(
    question: str,
    examples: str,
    reflections: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionReAct prompt template for the agent.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        reflections (str): Existing reflections.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = prompt.format(
        question=question,
        examples=examples,
        reflections=reflections,
        scratchpad=scratchpad,
        max_steps=max_steps,
        **additional_keys,
    )

    return prompt


def _prompt_react_agent(
    llm: BaseLLM,
    question: str,
    examples: str,
    reflections: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> Response:
    """Generates a ReAct prompt for thought and action.

    Used with ReflexionReAct.

    Args:
        llm (BaseLLM): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        reflections (str): Existing list of reflections.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        Response: The generated reflection prompt.
    """
    prompt = _build_react_agent_prompt(
        question=question,
        examples=examples,
        reflections=reflections,
        scratchpad=scratchpad,
        max_steps=max_steps,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    out = llm(prompt)
    return out

def _is_halted(
    finished: bool,
    step_idx: int,
    question: str,
    scratchpad: str,
    examples: str,
    reflections: str,
    max_steps: int,
    max_tokens: int,
    enc: Encoding,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on three conditions:
    completion (finished), exceeding maximum steps, or exceeding maximum token limit.
    The token limit is evaluated based on the encoded length of the prompt.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_idx (int): Current step number.
        question (str): The question being processed.
        scratchpad (str): The scratchpad content.
        examples (str): Fewshot examples.
        reflections (str): Reflections.
        max_steps (int): Maximum allowed steps.
        max_tokens (int): Maximum allowed token count.
        enc (Encoding): The encoder to calculate token length.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to

    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_idx > max_steps
    over_token_limit = (
        len(
            enc.encode(
                _build_react_agent_prompt(
                    examples=examples,
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=max_steps,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )
        )
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit

def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct and Reflexion.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument