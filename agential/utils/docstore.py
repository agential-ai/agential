"""Docstore and search-related logic."""

from abc import ABC, abstractmethod
from typing import List

import requests

from bs4 import BeautifulSoup


class BaseDocstoreExplorer(ABC):
    """Base class for docstore explorer."""

    @abstractmethod
    def search(self, term: str) -> str:
        """Search for a term in the docstore, and if found save.

        Args:
            term (str): The term to search for.

        Returns:
            str: The search result or observation, typically stored in self.obs.
        """
        raise NotImplementedError

    @abstractmethod
    def lookup(self, term: str) -> str:
        """Lookup a term in the docstore, and if found save.

        Args:
            term (str): The term to lookup.

        Returns:
            str: The lookup result or observation, typically stored in self.obs.
        """
        raise NotImplementedError


# Ref: https://github.com/ysymyth/ReAct/blob/master/wikienv.py
class DocstoreExplorer(BaseDocstoreExplorer):
    """Class to assist with exploration of a document store."""

    def __init__(self) -> None:
        """Initialize with a docstore, and set initial document to None."""
        self.page: str = ""
        self.lookup_keyword: str = ""
        self.lookup_list: List[str] = []
        self.lookup_cnt: int = 0
        self.obs: str = ""

    def clean_str(self, p: str) -> str:
        """Clean and decode a string.

        Args:
            p (str): The string to be cleaned and decoded.

        Return:
            str: The cleaned string.
        """
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

    def get_page_obs(self, page: str) -> str:
        """Retrieve the observation for a given page.

        Args:
            page (str): The page content to be processed.

        Returns:
            str: The observation derived from the page content.
        """
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")

        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return " ".join(sentences[:5])

    def construct_lookup_list(self, keyword: str) -> List[str]:
        """Constructs a list of paragraphs containing the given keyword.

        Args:
            keyword (str): The keyword to search for in the paragraphs.

        Returns:
            List[str]: A list of paragraphs that contain the keyword (case-insensitive).
        """
        if self.page is None:
            return []
        paragraphs = self.page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        sentences: List[str] = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]

        parts = sentences
        parts = [p for p in parts if keyword.lower() in p.lower()]
        return parts

    def search_step(self, entity: str) -> None:
        """Perform a search step for the given entity.

        This method prepares the entity string for a search operation by replacing spaces with plus signs.

        Args:
            entity (str): The entity to search for.
        """
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        response_text = requests.get(search_url).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            result_titles = [
                self.clean_str(div.get_text().strip()) for div in result_divs
            ]
            self.obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        else:
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                self.search_step("[" + entity + "]")
            else:
                self.page = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        self.page += self.clean_str(p)
                        if not p.endswith("\n"):
                            self.page += "\n"
                self.lookup_keyword = ""
                self.lookup_list = []
                self.lookup_cnt = 0
                self.obs = self.get_page_obs(self.page)

    def search(self, term: str) -> str:
        """Performs a search for the given term and updates the object's state.

        This method initiates a search process for the specified term by calling
        the search_step method. It updates the object's internal state based on
        the search results.

        Args:
            term (str): The search term to look up.

        Returns:
            str: The search result or observation, typically stored in self.obs.
        """
        self.search_step(term)
        return self.obs.strip()

    def lookup(self, term: str) -> str:
        """Perform a lookup operation for the given term.

        Args:
            term (str): The term to look up.

        Returns:
            str: The result of the lookup operation.
        """
        if self.lookup_keyword != term:
            self.lookup_keyword = term
            self.lookup_list = self.construct_lookup_list(term)
            self.lookup_cnt = 0
        if self.lookup_cnt >= len(self.lookup_list):
            self.obs = "No more results.\n"
        else:
            self.obs = (
                f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                + self.lookup_list[self.lookup_cnt]
            )
            self.lookup_cnt += 1

        return self.obs.strip()
