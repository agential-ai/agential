"""Unit tests for CoT Math strategies."""

from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_COT
from agential.core.llm import BaseLLM, MockLLM, Response
from agential.prompting.cot.output import CoTOutput, CoTStepOutput
from agential.prompting.cot.prompts import COT_INSTRUCTION_GSM8K
from agential.prompting.cot.strategies.math import (
    CoTGSM8KStrategy,
    CoTMathStrategy,
    CoTSVAMPStrategy,
    CoTTabMWPStrategy,
)


def test_init() -> None:
    """Tests the initialization of the CoTMathStrategy."""
    strategy = CoTMathStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    assert isinstance(strategy.llm, BaseLLM)


def test_generate() -> None:
    """Tests the generate method."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = "-9867630"

    gt_out = CoTOutput(answer='\n```python\n\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\n\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nmoney_per_egg = 2\nmoney_per_day = eggs_remaining * money_per_egg\nanswer = money_per_day\n\n```\n', total_prompt_tokens=20, total_completion_tokens=40, total_tokens=60, total_prompt_cost=3e-05, total_completion_cost=7.999999999999999e-05, total_cost=0.00010999999999999999, total_prompt_time=1.0, total_time=0.5, additional_info=[[CoTStepOutput(thought="Let's calculate the number of fresh duck eggs she has left after eating some and baking muffins. Then, we can determine how much money she makes by selling the remaining eggs at the farmers' market. We'll use the given information to calculate the daily earnings.", answer='\n```python\n\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\n\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nmoney_per_egg = 2\nmoney_per_day = eggs_remaining * money_per_egg\nanswer = money_per_day\n\n```\n', thought_response=Response(input_text='', output_text="Let's calculate the number of fresh duck eggs she has left after eating some and baking muffins. Then, we can determine how much money she makes by selling the remaining eggs at the farmers' market. We'll use the given information to calculate the daily earnings. \n\nAction: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\n\ntotal_eggs_after_use = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nmoney_made_per_egg = 2\nmoney_made_per_day = total_eggs_after_use * money_made_per_egg\nanswer = money_made_per_day\n```", prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5), answer_response=Response(input_text='', output_text='```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\n\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nmoney_per_egg = 2\nmoney_per_day = eggs_remaining * money_per_egg\nanswer = money_per_day\n```', prompt_tokens=10, completion_tokens=20, total_tokens=30, prompt_cost=1.5e-05, completion_cost=3.9999999999999996e-05, total_cost=5.4999999999999995e-05, prompt_time=0.5))]])      
    responses = [
        "Let's calculate the number of fresh duck eggs she has left after eating some and baking muffins. Then, we can determine how much money she makes by selling the remaining eggs at the farmers' market. We'll use the given information to calculate the daily earnings. \n\nAction: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\n\ntotal_eggs_after_use = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nmoney_made_per_egg = 2\nmoney_made_per_day = total_eggs_after_use * money_made_per_egg\nanswer = money_made_per_day\n```",
        '```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\n\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nmoney_per_egg = 2\nmoney_per_day = eggs_remaining * money_per_egg\nanswer = money_per_day\n```',
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strat = CoTMathStrategy(llm=llm, testing=True)
    out = strat.generate(
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        prompt=COT_INSTRUCTION_GSM8K,
        additional_keys={},
        num_retries=1,
        warming=[None],
    )
    assert out == gt_out


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = CoTGSM8KStrategy(llm=llm)
    svamp_strategy = CoTSVAMPStrategy(llm=llm)
    tabmwp_strategy = CoTTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, CoTGSM8KStrategy)
    assert isinstance(svamp_strategy, CoTSVAMPStrategy)
    assert isinstance(tabmwp_strategy, CoTTabMWPStrategy)
