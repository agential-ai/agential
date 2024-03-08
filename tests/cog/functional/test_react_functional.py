"""Unit tests for ReAct functional module."""
import tiktoken

from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.functional.react import (
    _build_agent_prompt,
    _is_halted,
    _prompt_agent,
    _check_keyword,
    _process_ob,
)
from discussion_agents.cog.prompts.react import (
    REACT_ALFWORLD_INSTRUCTION,
    REACT_ALFWORLD_PROMPTS_EXAMPLE,
    REACT_WEBTHINK_SIMPLE3_FEVER_EXAMPLES,
    REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
)


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(question="", scratchpad="")
    assert isinstance(prompt, str)


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]), question="", scratchpad=""
    )
    assert isinstance(out, str)
    assert out == "1"


def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
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


def test_check_keyword():
    alfworld_example = REACT_ALFWORLD_PROMPTS_EXAMPLE["react_put_0"]
    step_utilised = _check_keyword(alfworld_example)
    bool_list = [bool(item) for item in step_utilised]
    assert bool_list == [False, True, True]
    fever_example = REACT_WEBTHINK_SIMPLE3_FEVER_EXAMPLES
    step_utilised = _check_keyword(fever_example)
    bool_list = [bool(item) for item in step_utilised]
    assert bool_list == [True, True, True]
    hotpotqa_example = REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES
    step_utilised = _check_keyword(hotpotqa_example)
    bool_list = [bool(item) for item in step_utilised]
    assert bool_list == [True, True, True]


def test_process_ob():
    example_input = "You arrive at loc 22. On the countertop 2, you see a butterknife 1, a cellphone 1, a creditcard 1, a knife 1, a lettuce 1, a saltshaker 2, a saltshaker 1, a statue 1, and a tomato 1.\nYou pick up the tomato 1 from the countertop 2."
    example_output = _process_ob(example_input)
    expected_output = "On the countertop 2, you see a butterknife 1, a cellphone 1, a creditcard 1, a knife 1, a lettuce 1, a saltshaker 2, a saltshaker 1, a statue 1, and a tomato 1.\nYou pick up the tomato 1 from the countertop 2."
    assert example_output == expected_output
    example_input = "You arrive at loc 30. The fridge 1 is closed."
    example_output = _process_ob(example_input)
    expected_output = "The fridge 1 is closed."
    assert example_output == expected_output
