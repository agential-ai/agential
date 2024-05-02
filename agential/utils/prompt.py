"""Utility functions for prompting LangChain-provided LLMs."""

from typing import Dict

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage

def prompt_llm(
    llm: BaseChatModel,
    keys: Dict[str, str],
    prompt_template: str,
) -> str:
    """General function to prompt a large language model (LLM).

    Parameters:
        llm (BaseChatModel): The language model to use.
        keys (Dict[str, str]): Dictionary of keys and values to format the prompt template.
        prompt_template (str): The template string used to generate the prompt.

    Returns:
        str: The response from the language model.
    """
    # Use the prompt template and keys to generate the final prompt.
    try:
        formatted_prompt = PromptTemplate.from_template(prompt_template).format(**keys)
    except KeyError as e:
        raise ValueError(f"Missing key for prompt formatting: {e}")

    # Generate the response using the language model.
    response = llm(
        [
            HumanMessage(
                content=formatted_prompt,
            )
        ]
    ).content

    # Ensure the response is a string and trim it.
    if not isinstance(response, str):
        raise TypeError("Expected string response from language model")
    
    return response.strip()