"""Test the CLIN general strategy."""

import tiktoken
from agential.agents.clin.prompts import (
    CLIN_ADAPT_META_SUMMARY_SYSTEM,
    CLIN_ADAPT_SUMMARY_SYSTEM,
    CLIN_INSTRUCTION_HOTPOTQA,
)
from agential.agents.clin.strategies.general import CLINGeneralStrategy
from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_REACT
from agential.core.llm import MockLLM, Response


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
    gt_scratchpad = "\nThought 0: Thought: I need to find the capital of France."
    gt_thought = "Thought: I need to find the capital of France."
    gt_thought_response = Response(
        input_text="",
        output_text="Thought: I need to find the capital of France.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )
    llm = MockLLM(
        "gpt-3.5-turbo", responses=["Thought: I need to find the capital of France."]
    )
    strat = CLINGeneralStrategy(llm=llm)
    scratchpad, thought, thought_response = strat.generate_thought(
        idx=0,
        scratchpad="",
        question="What is the capital of France?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        summaries="",
        summary_system=CLIN_ADAPT_SUMMARY_SYSTEM,
        meta_summaries="",
        meta_summary_system=CLIN_ADAPT_META_SUMMARY_SYSTEM,
        prompt=CLIN_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert scratchpad == gt_scratchpad
    assert thought == gt_thought
    assert thought_response == gt_thought_response


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
