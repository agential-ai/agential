"""Test docstore-related logic."""

from unittest.mock import MagicMock

from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.documents.base import Document

from agential.utils.docstore import DocstoreExplorer


def test_docstore_explorer():
    """Tests DocstoreExplorer with manually mocked Wikipedia API."""
    # Create a mock Wikipedia docstore.
    wikipedia_docstore = MagicMock(spec=Wikipedia)

    # Create a mock document to be returned by the search method.
    mock_document = Document(
        page_content="Python is a programming language.\n\nIt is widely used."
    )

    # Set the mock search method to return the mock document.
    wikipedia_docstore.search.return_value = mock_document

    # Create an instance of DocstoreExplorer.
    explorer = DocstoreExplorer(wikipedia_docstore)

    # Test search functionality.
    search_term = "Python (programming language)"
    search_result = explorer.search(search_term)

    # Assert the search result is the summary of the document.
    assert search_result == "Python is a programming language."

    # Test lookup functionality.
    lookup_term = "programming"
    lookup_result = explorer.lookup(lookup_term)

    # Assert the lookup result is correct.
    assert "(Result 1/1) Python is a programming language." in lookup_result

    # Test lookup functionality with no results.
    lookup_term_no_result = "nonexistent"
    lookup_result_no_result = explorer.lookup(lookup_term_no_result)
    assert lookup_result_no_result == "No Results"

    # Test lookup functionality with no more results.
    lookup_term = "python"
    explorer.lookup(lookup_term)  # First call
    lookup_result_no_more = explorer.lookup(
        lookup_term
    )  # Second call, no more results.
    assert lookup_result_no_more == "No More Results"
