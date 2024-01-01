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