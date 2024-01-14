"""Unit tests for formatting functions."""
from datetime import datetime

import pytest

from langchain.schema.document import Document

from discussion_agents.utils.format import (
    format_memories_detail,
    format_memories_simple,
)


def test_format_memories_detail() -> None:
    """Test format_memories_detail."""
    test_date = datetime(year=2022, month=11, day=14, hour=3, minute=14)

    # Test formatting 1 Document.
    gt_doc_out = "-[Monday November 14, 2022 -- 03:14 AM]: Some page content."

    doc = Document(
        page_content="Some page content.", metadata={"created_at": test_date}
    )
    doc_out = format_memories_detail(memories=doc, prefix="-")
    assert doc_out == gt_doc_out

    # Test formatting multiple Documents.
    gt_docs_out = "> [Monday November 14, 2022 -- 03:14 AM]: Number 0.\n> [Monday November 14, 2022 -- 03:14 AM]: Number 1."

    docs = []
    for i in range(2):
        docs.append(
            Document(page_content=f"Number {i}.", metadata={"created_at": test_date})
        )

    docs_out = format_memories_detail(memories=docs, prefix="> ")
    assert docs_out == gt_docs_out

    # Test error raise.
    with pytest.raises(TypeError):
        doc = Document(page_content="test")
        _ = format_memories_detail(doc)


def test_format_memories_simple() -> None:
    """Test format_memories_simple."""
    # Test formatting 1 Document.
    gt_doc_out = "Some page content."

    doc = Document(
        page_content="Some page content.",
    )
    doc_out = format_memories_simple(relevant_memories=doc)
    assert doc_out == gt_doc_out

    # Test formatting multiple Documents.
    gt_docs_out = "Number 0.; Number 1."

    docs = []
    for i in range(2):
        docs.append(Document(page_content=f"Number {i}."))
    docs_out = format_memories_simple(relevant_memories=docs)
    assert docs_out == gt_docs_out