"""Unit tests for CoT general strategy."""

from agential.cog.cot.output import CoTOutput
from agential.cog.cot.strategies.general import CoTGeneralStrategy
from agential.llm.llm import BaseLLM, MockLLM, Response


def test_init() -> None:
    """Tests the initialization of the CoT general strategy."""
    strategy = CoTGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate() -> None:
    """Tests the generate method."""
    strategy = CoTGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=["1"]), testing=True
    )
    out = strategy.generate(
        question="What is the capital of France?",
        examples="",
        prompt="",
        additional_keys={},
    )
    assert out == CoTOutput(
        answer="1",
        total_prompt_tokens=10,
        total_completion_tokens=20,
        total_tokens=30,
        total_prompt_cost=1.5e-05,
        total_completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        total_prompt_time=0.5,
        total_time=0.5,
        additional_info=Response(
            input_text="",
            output_text="1",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
    )


def test_reset() -> None:
    """Tests the reset method."""
    strategy = CoTGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    strategy.reset()
