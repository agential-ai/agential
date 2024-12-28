"""Unit tests for ReAct math strategies."""

from tiktoken import Encoding

from agential.agents.react.prompts import REACT_INSTRUCTION_GSM8K
from agential.agents.react.strategies.math import (
    ReActGSM8KStrategy,
    ReActMathStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
)
from agential.core.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_REACT
from agential.core.llm import BaseLLM, MockLLM, Response


def test_init() -> None:
    """Test ReActMathStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActMathStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert strategy.max_steps == 6


def test_generate_action() -> None:
    """Tests ReActMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_scratchpad = "\nAction 0: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]"
    gt_query = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n"
    responses = [
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReActMathStrategy(llm=llm)
    scratchpad, action_type, query, action_response = strategy.generate_action(
        idx=0,
        scratchpad="",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert action_type == "Calculate"
    assert query == gt_query
    assert scratchpad == gt_scratchpad
    assert action_response == Response(
        input_text="",
        output_text="Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Tests ReActMathStrategy generate_observation."""
    # Test Calculate.
    gt_obs = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630"
    gt_scratchpad = "\nObservation 0: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630"
    gt_answer = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n"
    action_type = "Calculate"
    query = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActMathStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert obs == gt_obs
    assert scratchpad == gt_scratchpad
    assert answer == gt_answer
    assert not finished
    assert external_tool_info == {"execution_status": "Done", "code_answer": -9867630}

    # Test Finish.
    gt_obs = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```"
    gt_scratchpad = "\nObservation 0: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```"
    action_type = "Finish"
    query = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActMathStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert obs == gt_obs
    assert answer == query
    assert finished is True
    assert scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "Done", "code_answer": -9867630}

    # Test error case.
    gt_scratchpad = "\nObservation 0: Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
    action_type = "Unknown"
    query = "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n"
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReActMathStrategy(llm=llm)
    scratchpad, answer, obs, finished, external_tool_info = (
        strategy.generate_observation(
            idx=0, scratchpad="", action_type=action_type, query=query
        )
    )
    assert (
        obs == "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
    )
    assert answer == "\n```python\n\n```\n"
    assert finished is False
    assert scratchpad == gt_scratchpad
    assert external_tool_info == {"execution_status": "", "code_answer": ""}


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = ReActGSM8KStrategy(llm=llm)
    svamp_strategy = ReActSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReActTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReActGSM8KStrategy)
    assert isinstance(svamp_strategy, ReActSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReActTabMWPStrategy)
