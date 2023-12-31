"""Utility functions for formatting LangChain Documents, the base class for storing data."""
from typing import List, Union

from langchain.schema import Document


def format_memories_detail(
    memories: Union[Document, List[Document]], prefix: str = ""
) -> str:
    """Formats memories with created_at time and an optional prefix.

    Note: All memory documents must have a 'created_at' key within the
    metadata.

    Args:
        memories (Union[Document, List[Document]]): The memories to be formatted.
            It can be a single LangChain Document or a list of Document objects.
        prefix (str, optional): A prefix to be added before each formatted memory.
            Defaults to an empty string.

    Returns:
        str: A string containing the formatted memories with timestamps and prefix;
            newline-character delineated.

    Example:
        doc = Document(
            page_content="Some page content.", metadata={"created_at": test_date}
        )
        doc_out = format_memories_detail(memories=doc, prefix="-")
        # "-[Monday November 14, 2022 -- 03:14 AM]: Some page content."
    """
    if isinstance(memories, Document):
        memories = [memories]

    content = []
    for mem in memories:
        if isinstance(mem, Document):
            if "created_at" not in mem.metadata:
                raise TypeError(
                    "Input `memories` Document(s) must have 'created_at' key in metadata."
                )
            created_time = mem.metadata["created_at"].strftime(
                "%A %B %d, %Y -- %H:%M %p"
            )
            content.append(f"{prefix}[{created_time}]: {mem.page_content.strip()}")
    return "\n".join([f"{mem}" for mem in content])


def format_memories_simple(relevant_memories: Union[Document, List[Document]]) -> str:
    """Formats memories delineated by ';'.

    Args:
        relevant_memories (Union[Document, List[Document]]): The memories to be formatted.
            It can be a single LangChain Document or a list of Documents.

    Returns:
        str: A string containing the formatted memories separated by ';'.

    Example:
        docs = []
        for i in range(2):
            docs.append(Document(page_content=f"Number {i}."))
        docs_out = format_memories_simple(relevant_memories=docs)
        # "Number 0.; Number 1."
    """
    if isinstance(relevant_memories, Document):
        relevant_memories = [relevant_memories]
    return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

def clean_str(s: str) -> str:
    """Converts a string with mixed encoding to proper UTF-8 format.

    This function takes a string `s` that may contain Unicode escape sequences and/or Latin-1 encoded characters. 
    It processes the string to interpret Unicode escape sequences and correct any Latin-1 encoded parts, returning the string in UTF-8 format.

    Args:
        s (str): The input string potentially containing Unicode escape sequences and Latin-1 encoded characters.

    Returns:
        str: The UTF-8 encoded string with properly interpreted characters.

    Note:
        This function assumes that the input string is a mix of UTF-8 encoded characters and Unicode escape sequences. 
        It may not work as intended if the input string has a different encoding or if it contains characters outside the Latin-1 range.
    """
    return s.encode().decode("unicode-escape").encode("latin1").decode("utf-8")