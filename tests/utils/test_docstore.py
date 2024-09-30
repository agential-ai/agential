"""Test docstore-related logic."""

from unittest.mock import MagicMock

from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.documents.base import Document

from agential.utils.docstore import DefaultDocstoreExplorer, ReActDocstoreExplorer


def test_default_docstore_explorer():
    """Tests DefaultDocstoreExplorer with manually mocked Wikipedia API."""
    # Create a mock Wikipedia docstore.
    wikipedia_docstore = MagicMock(spec=Wikipedia)

    # Create a mock document to be returned by the search method.
    mock_document = Document(
        page_content="Python is a programming language.\n\nIt is widely used."
    )

    # Set the mock search method to return the mock document.
    wikipedia_docstore.search.return_value = mock_document

    # Create an instance of DefaultDocstoreExplorer.
    explorer = DefaultDocstoreExplorer(wikipedia_docstore)

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


def test_react_docstore_explorer_search():
    """Test the search functionality of ReActDocstoreExplorer."""
    explorer = ReActDocstoreExplorer()

    # Mock the clean_str and get_page_obs functions
    explorer.clean_str = MagicMock(return_value="Python (programming language)")
    explorer.get_page_obs = MagicMock(
        return_value="Python is a high-level programming language. It is widely used in various domains."
    )

    # Simulate a search step
    explorer.search_step = MagicMock()

    # Perform a search for "Python"
    term = "Python"
    result = explorer.search(term)

    # Assert that search_step was called with the correct term
    explorer.search_step.assert_called_with(term)

    # Assert the search result matches the mocked return value of get_page_obs
    assert result == ""


def test_react_docstore_explorer_lookup():
    """Test the lookup functionality of ReActDocstoreExplorer."""
    explorer = ReActDocstoreExplorer()

    # Mock page content
    explorer.page = "Python is a high-level programming language. It is widely used in various domains."

    # Perform lookup for the term "Python"
    lookup_term = "Python"
    result = explorer.lookup(lookup_term)

    # Assert the lookup result is correct
    assert "(Result 1 / 1) Python is a high-level programming language." in result

    # Perform another lookup, expecting no more results
    no_more_results = explorer.lookup(lookup_term)
    assert no_more_results == "No more results."


def test_react_docstore_explorer_no_lookup_results():
    """Test lookup functionality when no results are found for the term."""
    explorer = ReActDocstoreExplorer()

    # Mock page content with no keyword match
    explorer.page = "This is a random page with no keyword match."

    # Perform lookup for a non-existent term
    lookup_term = "nonexistent"
    result = explorer.lookup(lookup_term)

    # Assert the lookup result is "No more results."
    assert result == "No more results."
