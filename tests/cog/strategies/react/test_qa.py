"""Unit tests for ReAct QA strategies."""
from tiktoken import Encoding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.react.base import DocstoreExplorer

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.strategies.react.qa import ReActQAStrategy
from agential.cog.prompts.agents.react import (
    REACT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmarks.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT


def test_init() -> None:
    """Test ReActQAStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 3896
    assert isinstance(strategy.docstore, DocstoreExplorer)
    assert isinstance(strategy.enc, Encoding)
    assert strategy._scratchpad == ""
    assert strategy._finished == False


def test_generate() -> None:
    """Tests ReActQAStrategy generate."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_result = 'I need to search for the best kickboxer in the world who has been involved in controversies and crimes.'
    gt_scratchpad = '\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes.'
    responses = [
        "I need to search for the best kickboxer in the world who has been involved in controversies and crimes.\nAction 1: Search[best kickboxer in the world controversies crimes]\nObservation 1: Could not find exact match. Similar: ['List of kickboxers', 'Kickboxing', 'List of controversies involving Kickboxing']\nThought 2: I should try searching for the best kickboxer in the world and then look for any controversies or crimes related to him.\nAction 2: Search[best kickboxer in the world]\nObservation 2: Could not find exact match. Similar: ['List of best kickboxers in the world', 'List of kickboxing organizations', 'Kickboxing', 'Best Fighters in the World']\nThought 3: I can try searching for top kickboxers and then look for controversies and crimes.\nAction 3: Search[top kickboxers]\nObservation 3: Could not find exact match. Similar: ['Top 10 kickboxers', 'Top 5 kickboxers', 'Top 15 kickboxers']\nThought 4: I need to refine my search terms to find the information I need.\nAction 4: Search[most famous kickboxer controversies crimes]\nObservation 4: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 5: I should try searching for famous kickboxers involved in controversies and crimes.\nAction 5: Search[famous kickboxers controversies crimes]\nObservation 5: Could not find exact match. Similar: ['Famous kickboxers', 'Kickboxing controversies', 'Famous kickboxers in the world']\nThought 6: I am unable to find the specific information I need within the given steps. \nAction 6: Finish[unable to find answer]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    result = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert result == gt_result
    assert not strategy._finished
    assert strategy._scratchpad == gt_scratchpad


def test_generate_action() -> None:
    """Tests ReActQAStrategy generate_action."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_action_type = "Search"
    gt_query = 'best kick boxer in the world controversies crimes'
    init_scratchpad = '\nThought: I need to search for the best kickboxer in the world who has been involved in controversies and crimes.'
    responses = [
        'Search[best kick boxer in the world controversies crimes]'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReActQAStrategy(llm=llm)
    strategy._scratchpad = init_scratchpad
    strategy._finished = False
    action_type, query = strategy.generate_action(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == gt_action_type
    assert query == gt_query


def test_generate_observation() -> None:
    """Tests ReActQAStrategy generate_observation."""

def test_create_output_dict() -> None:
    """Tests ReActQAStrategy create_output_dict."""

def test_halting_condition() -> None:
    """Tests ReActQAStrategy halting_condition."""

def test_reset() -> None:
    """Tests ReActQAStrategy reset."""

def test_instantiate_strategies() -> None:
    """Test instantiate all QA strategies."""