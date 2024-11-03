"""Test the CLIN QA strategy."""

import tiktoken

from agential.agents.clin.prompts import CLIN_ADAPT_META_SUMMARY_SYSTEM, CLIN_ADAPT_SUMMARY_SYSTEM, CLIN_INSTRUCTION_HOTPOTQA
from agential.agents.clin.strategies.qa import CLINQAStrategy
from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.core.llm import MockLLM, Response
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
    """Tests CLIN QA strategy generate action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nAction 1: Search[VIVA Media AG]"
    responses = [
        "Search[VIVA Media AG]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = CLINQAStrategy(llm=llm)
    scratchpad, action_type, query, thought_response = strategy.generate_action(
        idx=1,
        scratchpad="",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        summaries="",
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summaries="",
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == "Search"
    assert query == "VIVA Media AG"
    assert scratchpad == gt_scratchpad
    assert thought_response == Response(
        input_text="",
        output_text="Search[VIVA Media AG]",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Tests CLIN QA strategy generate observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = CLINQAStrategy(llm=llm)
    strategy.docstore.search = lambda x: "Search result"
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Search",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "Search result", "lookup_result": ""}
    assert scratchpad == "\nObservation 1: Search result"
    assert answer == ""
    assert not finished

    strategy.docstore.lookup = lambda x: "Lookup result"
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Lookup",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "", "lookup_result": "Lookup result"}
    assert scratchpad == "\nObservation 1: Lookup result"
    assert answer == ""
    assert not finished

    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=1,
            scratchpad="",
            action_type="Finish",
            query="VIVA Media AG",
            key="key1",
        )
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert external_tool_info == {"search_result": "", "lookup_result": ""}
    assert scratchpad == "\nObservation 1: Answer is INCORRECT"
    assert answer == "VIVA Media AG"
    assert finished


def test_halting_condition() -> None:
    """Test CLIN QA strategy halting condition."""
    strategy = CLINQAStrategy(llm=None, memory=None)
    assert strategy.halting_condition(
        idx=0,
        key="",
        answer="",
    ) is True