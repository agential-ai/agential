"""Unit tests for the ReAct general strategy."""

import pytest

from tiktoken.core import Encoding

from agential.cog.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.cog.react.prompts import (
    REACT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.react.strategies.general import ReActGeneralStrategy
from agential.llm.llm import BaseLLM, MockLLM


def test_init() -> None:
    """Test ReActGeneralStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActGeneralStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert isinstance(strategy.enc, Encoding)


def test_generate_thought() -> None:
    """Tests ReActGeneralStrategy generate_thought."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_thought = "I need to search for the best kickboxer in the world who has been involved in controversies and crimes."
    gt_scratchpad = "\nThought 0: I need to search for the best kickboxer in the world who has been involved in controversies and crimes."
    gt_out = "I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction 1: Search[best kickboxer in the world controversies crimes]\nObservation 1: Could not find exact match. Similar: ['List of kickboxers', 'Kickboxing', 'List of controversies involving Kickboxing']\nThought 2: I should try searching for the best kickboxer in the world and then look for any controversies or crimes related to him.\nAction 2: Search[best kickboxer in the world]\nObservation 2: Could not find exact match. Similar: ['List of best kickboxers in the world', 'List of kickboxing organizations', 'Kickboxing', 'Best Fighters in the World']\nThought 3: I can try searching for top kickboxers and then look for controversies and crimes.\nAction 3: Search[top kickboxers]\nObservation 3: Could not find exact match. Similar: ['Top 10 kickboxers', 'Top 5 kickboxers', 'Top 15 kickboxers']\nThought 4: I need to refine my search terms to find the information I need.\nAction 4: Search[most famous kickboxer controversies crimes]\nObservation 4: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 5: I should try searching for famous kickboxers involved in controversies and crimes.\nAction 5: Search[famous kickboxers controversies crimes]\nObservation 5: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 6: I am unable to find the specific information I need within the given steps. \nAction 6: Finish[unable to find answer]"
    responses = [
        "I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction 1: Search[best kickboxer in the world controversies crimes]\nObservation 1: Could not find exact match. Similar: ['List of kickboxers', 'Kickboxing', 'List of controversies involving Kickboxing']\nThought 2: I should try searching for the best kickboxer in the world and then look for any controversies or crimes related to him.\nAction 2: Search[best kickboxer in the world]\nObservation 2: Could not find exact match. Similar: ['List of best kickboxers in the world', 'List of kickboxing organizations', 'Kickboxing', 'Best Fighters in the World']\nThought 3: I can try searching for top kickboxers and then look for controversies and crimes.\nAction 3: Search[top kickboxers]\nObservation 3: Could not find exact match. Similar: ['Top 10 kickboxers', 'Top 5 kickboxers', 'Top 15 kickboxers']\nThought 4: I need to refine my search terms to find the information I need.\nAction 4: Search[most famous kickboxer controversies crimes]\nObservation 4: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 5: I should try searching for famous kickboxers involved in controversies and crimes.\nAction 5: Search[famous kickboxers controversies crimes]\nObservation 5: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 6: I am unable to find the specific information I need within the given steps. \nAction 6: Finish[unable to find answer]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActGeneralStrategy(llm=llm)
    scratchpad, thought, out = strategy.generate_thought(
        idx=0,
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert scratchpad == gt_scratchpad
    assert thought == gt_thought
    assert out.choices[0].message.content == gt_out


def test_generate_action() -> None:
    """Test generate_action."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        _ = strategy.generate_action(
            idx=0,
            scratchpad="",
            question="What is the capital of France?",
            examples="Example content",
            prompt="Prompt content",
            additional_keys={},
        )


def test_generate_observation() -> None:
    """Test generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActGeneralStrategy(llm=llm)

    with pytest.raises(NotImplementedError):
        _ = strategy.generate_observation(
            idx=0,
            scratchpad="",
            action_type="Search",
            query="What is the capital of France?",
        )


def test_halting_condition() -> None:
    """Tests ReActGeneralStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActGeneralStrategy(llm=llm)
    finished = False
    idx = 0
    question = "What is the capital of France?"
    scratchpad = ""
    examples = ""
    prompt = "Answer the question."

    assert not strategy.halting_condition(
        finished, idx, question, scratchpad, examples, prompt, {}
    )


def test_reset() -> None:
    """Tests ReActGeneralStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActGeneralStrategy(llm=llm)

    strategy.reset()
    assert isinstance(strategy, ReActGeneralStrategy)
