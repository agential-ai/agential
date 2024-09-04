"""Unit tests for Standard general strategy."""

from agential.core.fewshots.hotpotqa import HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT
from agential.core.llm import BaseLLM, MockLLM, Response
from agential.prompting.standard.output import StandardOutput, StandardStepOutput
from agential.prompting.standard.prompts import STANDARD_INSTRUCTION_HOTPOTQA
from agential.prompting.standard.strategies.general import StandardGeneralStrategy


def test_init() -> None:
    """Tests the initialization of the standard prompting general strategy."""
    strategy = StandardGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate() -> None:
    """Tests the generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    gt_out = StandardOutput(
        answer="Badr Hari",
        total_prompt_tokens=10,
        total_completion_tokens=20,
        total_tokens=30,
        total_prompt_cost=1.5e-05,
        total_completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        total_prompt_time=0.5,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer="Badr Hari",
                    answer_response=Response(
                        input_text="",
                        output_text="Badr Hari",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                )
            ]
        ],
    )
    responses = [
        "Badr Hari",
        "Badr Hari",
    ]
    strategy = StandardGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        key="Badr Hari",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        num_retries=1,
        warming=[None, 0.123, None, 0.2],
    )
    assert out == gt_out

    # Test num_retries=2.
    gt_out = StandardOutput(
        answer="Paris",
        total_prompt_tokens=10,
        total_completion_tokens=20,
        total_tokens=30,
        total_prompt_cost=1.5e-05,
        total_completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        total_prompt_time=0.5,
        total_time=0.5,
        additional_info=[
            [
                StandardStepOutput(
                    answer="Paris",
                    answer_response=Response(
                        input_text="",
                        output_text="Paris",
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                )
            ]
        ],
    )
    responses = [
        "Paris",
        "Paris",
        "Paris",
        "Paris",
    ]
    strategy = StandardGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        key="Paris",
        question="What is the capital of France?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        num_retries=2,
        warming=[None, 0.123, None, 0.2],
    )

    assert out == gt_out


def test_reset() -> None:
    """Tests the reset method."""
    strategy = StandardGeneralStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    strategy.reset()
