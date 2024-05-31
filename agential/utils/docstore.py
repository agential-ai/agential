"""Docstore-related logic."""

from typing import Optional, List
from langchain_core.documents.base import Document
from langchain_community.docstore.base import Docstore

# Ref: https://github.com/langchain-ai/langchain/blob/0214246dc69dd2d4e11fd567308f666c220cfb0d/libs/langchain/langchain/agents/react/base.py#L72
class DocstoreExplorer:
    """Class to assist with exploration of a document store."""

    def __init__(self, docstore: Docstore):
        """Initialize with a docstore, and set initial document to None."""
        self.docstore = docstore
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save."""
        result = self.docstore.search(term)
        if isinstance(result, Document):
            self.document = result
            return self._summary
        else:
            self.document = None
            return result

    def lookup(self, term: str) -> str:
        """Lookup a term in document (if saved)."""
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _summary(self) -> str:
        return self._paragraphs[0]

    @property
    def _paragraphs(self) -> List[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")