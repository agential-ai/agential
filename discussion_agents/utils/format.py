"""Utility functions for formatting LangChain Documents, the base class for storing data."""

from typing import Union, List
from langchain.schema import Document

def format_memories_detail(
    relevant_memories: Union[Document, List[Document]], prefix: str = ""
) -> str:
    """Formats memories with created_at time and an optional prefix.

    Args:
        relevant_memories (Union[Document, List[Document]]): The memories to be formatted.
            It can be a single Document or a list of Document objects.
        prefix (str, optional): A prefix to be added before each formatted memory.
            Defaults to an empty string.

    Returns:
        str: A string containing the formatted memories with timestamps and prefix;
            newline-character delineated.
    """
    if isinstance(relevant_memories, Document):
        relevant_memories = [relevant_memories]

    content = []
    for mem in relevant_memories:
        if isinstance(mem, Document):
            created_time = mem.metadata["created_at"].strftime(
                "%A %B %d, %Y -- %H:%M %p"
            )
            content.append(f"{prefix}[{created_time}]: {mem.page_content.strip()}")
    return "\n".join([f"{mem}" for mem in content])

def format_memories_simple(
    relevant_memories: Union[Document, List[Document]]
) -> str:
    r"""Formats memories delineated by \';\'.

    Args:
        relevant_memories (Union[Document, List[Document]]): The memories to be formatted.
            It can be a single Document or a list of Documents.

    Returns:
        str: A string containing the formatted memories separated by \';\'.
    """
    if isinstance(relevant_memories, Document):
        relevant_memories = [relevant_memories]
    return "; ".join([f"{mem.page_content}" for mem in relevant_memories])
