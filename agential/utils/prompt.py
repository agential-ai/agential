"""Utility functions for prompting LangChain-provided LLMs."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage

def prompt_llm(
    llm: BaseChatModel,
    prompt: str,
) -> str:
    """General function to prompt a large language model (LLM).

    Parameters:
        llm (BaseChatModel): The language model to use.
        prompt (str): The content to prompt the language model.

    Returns:
        str: The response from the language model.
    """
    # Generate the response using the language model.
    response = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content

    # Ensure the response is a string and trim it.
    if not isinstance(response, str):
        raise TypeError("Expected string response from language model")
    
    return response.strip()