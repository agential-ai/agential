"""Functional module for ReAct."""
from typing import Any, Tuple, Dict
from tiktoken import Encoding
from langchain.prompts import PromptTemplate
from discussion_agents.cog.prompts.react import REACT_INSTRUCTION, WEBTHINK_SIMPLE6
from discussion_agents.utils.parse import remove_newline
from langchain.agents.react.base import DocstoreExplorer


def _build_agent_prompt(question: str, scratchpad: str) -> PromptTemplate:
    """Constructs a prompt template for the agent.

    This function formats a predefined prompt template (REACT_INSTRUCTION) with examples,
    the provided question, and a scratchpad.

    Args:
        question (str): The question to be included in the prompt.
        scratchpad (str): Additional scratchpad information to be included.

    Returns:
        PromptTemplate: A formatted prompt template ready for use.
    """
    prompt = PromptTemplate.from_template(REACT_INSTRUCTION).format(
        examples=WEBTHINK_SIMPLE6,
        question=question,
        scratchpad=scratchpad
    )
    return prompt

def _prompt_agent(llm: Any, question: str, scratchpad: str) -> str:
    """Generates a response from the LLM based on a given question and scratchpad.

    This function creates a prompt using `_build_agent_prompt` and then gets the LLM's
    output. The newline characters in the output are removed before returning.

    Args:
        llm (Any): The language model to be prompted.
        question (str): The question to ask the language model.
        scratchpad (str): Additional context or information for the language model.

    Returns:
        str: The processed response from the language model.
    """
    prompt = _build_agent_prompt(question=question, scratchpad=scratchpad)
    out = llm(prompt)
    return remove_newline(out)

def _is_halted(
    finished: bool,
    step_n: int, 
    max_steps: int, 
    question: str, 
    scratchpad: str, 
    max_tokens: int,
    enc: Encoding
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

    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_n > max_steps
    over_token_limit = len(enc.encode(_build_agent_prompt(question=question, scratchpad=scratchpad))) > max_tokens
    return finished or over_max_steps or over_token_limit


def react_think(llm: Any, question: str, scratchpad: str) -> str:
    """Generates a 'thought' based on the current question and scratchpad content.

    This function appends a thought to the scratchpad by using the _prompt_agent function
    to generate a response from the language model (LLM). The thought is extracted from
    the response before the 'Action' segment.

    Args:
        llm (Any): The language model to be prompted.
        question (str): The current question being considered.
        scratchpad (str): The existing scratchpad content.

    Returns:
        str: The updated scratchpad including the generated thought.
    """
    scratchpad += f"\nThought:"
    thought = _prompt_agent(
        llm=llm, 
        question=question, 
        scratchpad=scratchpad,
    ).split("Action")[0]
    scratchpad += " " + thought
    return scratchpad

def react_act(llm: Any, question: str, scratchpad: str) -> Tuple[str, str]:
    """Determines an action based on the current question and scratchpad content.

    This function appends an action to the scratchpad by using the _prompt_agent function
    to get a response from the language model (LLM). The action is extracted from
    the response before the 'Observation' segment.

    Args:
        llm (Any): The language model to be prompted.
        question (str): The current question being considered.
        scratchpad (str): The existing scratchpad content.

    Returns:
        Tuple[str, str]: A tuple containing the updated scratchpad and the determined action.
    """
    scratchpad += f"\nAction:"
    action = _prompt_agent(
        llm=llm, 
        question=question, 
        scratchpad=scratchpad,
    ).split("Observation")[0]
    scratchpad += " " + action
    return scratchpad, action

def react_observe(
    action_type: str, 
    query: str, 
    scratchpad: str, 
    step_n: int, 
    docstore: DocstoreExplorer
) -> Dict[str, Any]:
    """Updates the scratchpad based on the given action and query, and manages document store interactions.

    Depending on the action type ('finish', 'search', or 'lookup'), this function updates
    the scratchpad with the appropriate observation. It handles document store searches
    and lookups, and marks the process as finished if required.

    Args:
        action_type (str): The type of action to be taken ('finish', 'search', 'lookup').
        query (str): The query for the action, e.g., a search term or an answer.
        scratchpad (str): The current scratchpad content.
        step_n (int): The current step number in the process.
        docstore (DocstoreExplorer): The document store explorer for handling searches and lookups.

    Returns:
        Dict[str, Any]: A dictionary containing the updated scratchpad, answer (if any), 
                        a finished flag, and the updated step number.
    """
    scratchpad += f"\nObservation {step_n}: "
    
    out = {
        "scratchpad": scratchpad,
        "answer": None,
        "finished": False,
        "step_n": step_n
    }

    if action_type.lower() == "finish":
        out["answer"] = query
        out["finished"] = True
        out["scratchpad"] += f"{query}"
    elif action_type.lower() == "search":
        try:
            out["scratchpad"] += remove_newline(docstore.search(query))
        except Exception:
            out["scratchpad"] += f"Could not find that page, please try again."
        
    elif action_type.lower() == "lookup":
        try:
            out["scratchpad"] += remove_newline(docstore.lookup(query))
        except ValueError:
            out["scratchpad"] += f"The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
    else:
        out["scratchpad"] += "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."

    out["step_n"] += 1

    return out