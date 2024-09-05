"""Unit tests for standard prompting Math strategies."""

from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_DIRECT
from agential.core.llm import MockLLM, Response
from agential.prompting.standard.output import StandardOutput, StandardStepOutput
from agential.prompting.standard.prompts import STANDARD_INSTRUCTION_GSM8K
from agential.prompting.standard.strategies.general import StandardGeneralStrategy
from agential.prompting.standard.strategies.math import (
    StandardGSM8KStrategy,
    StandardMathStrategy,
    StandardSVAMPStrategy,
    StandardTabMWPStrategy,
)


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = StandardGSM8KStrategy(llm=llm)
    svamp_strategy = StandardSVAMPStrategy(llm=llm)
    tabmwp_strategy = StandardTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, StandardGSM8KStrategy)
    assert isinstance(svamp_strategy, StandardSVAMPStrategy)
    assert isinstance(tabmwp_strategy, StandardTabMWPStrategy)


def test_generate() -> None:
    """Tests the generate method."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_out = StandardOutput(
        answer="-9867630",
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
                    answer="-9867630",
                    answer_response=Response(
                        input_text="",
                        output_text=" -9867630",
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
        " -9867630",
    ]
    strategy = StandardMathStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        key=-9867630,
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_GSM8K,
        additional_keys={},
        num_retries=1,
        warming=[None],
    )
    assert out == gt_out

    # Test num_retries=2.
    gt_out = StandardOutput(
        answer="-9867630",
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
                    answer="-9867630",
                    answer_response=Response(
                        input_text="",
                        output_text="-9867630",
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
        "-9867630",
    ]
    strategy = StandardGeneralStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), testing=True
    )
    out = strategy.generate(
        question=question,
        key="-9867630",
        examples=GSM8K_FEWSHOT_EXAMPLES_DIRECT,
        prompt=STANDARD_INSTRUCTION_GSM8K,
        additional_keys={},
        num_retries=2,
        warming=[None, 0.123, None, 0.2],
    )
    assert out == gt_out
