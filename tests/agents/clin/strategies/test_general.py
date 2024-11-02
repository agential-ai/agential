"""Test the CLIN general strategy."""

import tiktoken
from agential.agents.clin.strategies.general import CLINGeneralStrategy

def test_init() -> None:
    """Test CLIN general strategy initialization."""
    strategy = CLINGeneralStrategy(llm=None, memory=None)
    assert strategy.max_trials == 3
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert strategy.enc == tiktoken.encoding_for_model("gpt-3.5-turbo")
    assert strategy.testing is False


def test_generate_thought() -> None:
    """Test CLIN general strategy generate thought."""


def test_generate_summary() -> None:
    """Test CLIN general strategy generate summary."""
    pass


def test_generate_meta_summary() -> None:
    """Test CLIN general strategy generate meta summary."""
    pass


def test_react_halting_condition() -> None:
    """Test CLIN general strategy react halting condition."""
    pass


def test_reset() -> None:
    """Test CLIN general strategy reset."""
    pass