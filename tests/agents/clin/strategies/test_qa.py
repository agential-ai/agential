"""Test the CLIN QA strategy."""

import tiktoken

from agential.agents.clin.strategies.qa import CLINQAStrategy
from agential.core.llm import MockLLM
from agential.utils.docstore import DocstoreExplorer


def test_init() -> None:
    """Test CLIN QA strategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINQAStrategy(llm=llm, memory=None)
    assert strategy.max_trials == 3
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert strategy.enc == tiktoken.encoding_for_model("gpt-3.5-turbo")
    assert strategy.testing is False
    assert isinstance(strategy.docstore, DocstoreExplorer)


def test_generate() -> None:
    """Test CLIN QA strategy generate."""


def test_generate_react() -> None:
    """Test CLIN QA strategy generate react."""


def test_generate_action() -> None:
    """Test CLIN QA strategy generate action."""


def test_generate_observation() -> None:
    """Test CLIN QA strategy generate observation."""


def test_halting_condition() -> None:
    """Test CLIN QA strategy halting condition."""
    strategy = CLINQAStrategy(llm=None, memory=None)
    assert strategy.halting_condition(
        idx=0,
        key="",
        answer="",
    ) is True