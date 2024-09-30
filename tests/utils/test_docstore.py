"""Test docstore-related logic."""

from unittest.mock import MagicMock

from agential.utils.docstore import DocstoreExplorer


def test_docstore_explorer_search():
    """Test the search functionality of DocstoreExplorer."""
    explorer = DocstoreExplorer()

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


def test_docstore_explorer_lookup():
    """Test the lookup functionality of DocstoreExplorer."""
    explorer = DocstoreExplorer()

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


def test_docstore_explorer_no_lookup_results():
    """Test lookup functionality when no results are found for the term."""
    explorer = DocstoreExplorer()

    # Mock page content with no keyword match
    explorer.page = "This is a random page with no keyword match."

    # Perform lookup for a non-existent term
    lookup_term = "nonexistent"
    result = explorer.lookup(lookup_term)

    # Assert the lookup result is "No more results."
    assert result == "No more results."
