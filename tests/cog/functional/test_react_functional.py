"""Unit tests for ReAct functional module."""
import tiktoken
from langchain.llms.fake import FakeListLLM
from langchain_community.docstore.wikipedia import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

from discussion_agents.cog.functional.react import (
    _build_agent_prompt,
    _prompt_agent,
    _is_halted,
    react_think,
    react_act,
    react_observe
)

def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(question="", scratchpad="")
    assert isinstance(prompt, str)

def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(llm=FakeListLLM(responses=["1"]), question="", scratchpad="")
    assert isinstance(out, str)
    assert out == "1"

def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model(
        "gpt-3.5-turbo"
    )
    assert _is_halted(True, 1, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test when step_n exceeds max_steps.
    assert _is_halted(False, 11, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test when encoded prompt exceeds max_tokens.
    assert _is_halted(False, 1, 10, "question", "scratchpad", 10, gpt3_5_turbo_enc)

    # Test when none of the conditions for halting are met.
    assert _is_halted(False, 1, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test edge case when step_n equals max_steps.
    assert _is_halted(False, 10, 10, "question", "scratchpad", 100, gpt3_5_turbo_enc)

    # Test edge case when encoded prompt equals max_tokens.
    assert _is_halted(False, 1, 10, "question", "scratchpad", 20, gpt3_5_turbo_enc)


def test_react_think() -> None:
    """Test react_think function."""
    out = react_think(
        llm=FakeListLLM(responses=["1"]),
        question="",
        scratchpad=""
    )
    assert out == "\nThought: 1"


def test_react_act() -> None:
    """Test react_act function."""
    out, action = react_act(
        llm=FakeListLLM(responses=["1"]),
        question="",
        scratchpad=""
    )
    assert out == "\nAction: 1"
    assert action == "1"


def test_react_observe() -> None:
    """Test react_observe function."""
    # Invalid action.
    gt = {
        'scratchpad': '\nObservation 0: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].',
        'answer': None,
        'finished': False,
        'step_n': 1
    }
    out = react_observe(
        action_type="invalid input",
        query="",
        scratchpad="",
        step_n=0,
        docstore=None
    )
    assert out == gt

    # Finish.
    gt = {
        'scratchpad': '\nObservation 0: ',
        'answer': '',
        'finished': True,
        'step_n': 1
    }
    out = react_observe(
        action_type="finish",
        query="",
        scratchpad="",
        step_n=0,
        docstore=None
    )
    assert out == gt

    # Search empty query.
    gt = {
        'scratchpad': '\nObservation 0: Could not find that page, please try again.',
        'answer': None,
        'finished': False,
        'step_n': 1
    }
    out = react_observe(
        action_type="search",
        query="",
        scratchpad="",
        step_n=0,
        docstore=DocstoreExplorer(Wikipedia())
    )
    assert out == gt

    # Search non-empty query.
    out = react_observe(
        action_type="search",
        query="deep learning",
        scratchpad="",
        step_n=0,
        docstore=DocstoreExplorer(Wikipedia())
    )
    assert out["step_n"] == 1
    assert not out["finished"]
    assert not out["answer"]
    assert "\nObservation 0: Could not find that page, please try again." not in out["scratchpad"]

    # Lookup empty query.
    gt = {
        'scratchpad': '\nObservation 0: The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.',
        'answer': None,
        'finished': False,
        'step_n': 1
    }
    out = react_observe(
        action_type="lookup",
        query="deep learning",
        scratchpad="",
        step_n=0,
        docstore=DocstoreExplorer(Wikipedia())
    )
    assert out == gt

    # Lookup non-empty query.
    docstore = DocstoreExplorer(Wikipedia())
    _ = react_observe(
        action_type="search",
        query="deep learning",
        scratchpad="",
        step_n=0,
        docstore=docstore
    )
    out = react_observe(
        action_type="lookup",
        query="deep",
        scratchpad="",
        step_n=0,
        docstore=docstore
    )
    gt = '\nObservation 0: The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
    assert out["step_n"] == 1
    assert not out["finished"]
    assert not out["answer"]
    assert gt not in out["scratchpad"]