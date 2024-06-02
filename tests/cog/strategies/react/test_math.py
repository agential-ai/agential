"""Unit tests for ReAct math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding
from agential.cog.prompts.agent.react import REACT_INSTRUCTION_GSM8K
from agential.cog.prompts.benchmark.gsm8k import GSM8K_FEWSHOT_EXAMPLES_REACT
from agential.cog.strategies.react.math import (
    ReActMathStrategy,
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
    parse_math_action,
)


def test_parse_math_action() -> None:
    """Test parse_math_action."""
    test_cases = [
        {
            "input": "Calculate[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Calculate", "def add(a, b): return a + b"),
        },
        {
            "input": "Finish[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Finish", "assert add(2, 3) == 5"),
        },
        {
            "input": "Finish[```python\nThe function is complete.\n```]",
            "expected": ("Finish", "The function is complete."),
        },
        {
            "input": "calculate[```python\ndef subtract(a, b): return a - b\n```]",
            "expected": ("Calculate", "def subtract(a, b): return a - b"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Calculate[```python\nassert subtract(5, 3) == 2\n```]",
            "expected": ("Calculate", "assert subtract(5, 3) == 2"),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
        {
            "input": "Finish[```python\n \n```]",
            "expected": ("Finish", ""),
        },
        {
            "input": "Calculate[```python\nfor i in range(10):\n    print(i)\n```]",
            "expected": ("Calculate", "for i in range(10):\n    print(i)"),
        },
    ]

    for case in test_cases:
        result = parse_math_action(case["input"])
        assert result == case["expected"]


def test_init() -> None:
    """Test ReActMathStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActMathStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 3896
    assert isinstance(strategy.enc, Encoding)
    assert strategy._current_answer == ""
    assert strategy._scratchpad == ""
    assert strategy._finished == False


def test_generate() -> None:
    """Tests ReActMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    gt_scratchpad = "\nThought: First, I need to calculate how many eggs Janet has left after eating three for breakfast and using the rest for muffins. Then, I can find out how much money she makes selling the remaining eggs at the market. Let's break this down step by step."
    gt_out = "First, I need to calculate how many eggs Janet has left after eating three for breakfast and using the rest for muffins. Then, I can find out how much money she makes selling the remaining eggs at the market. Let's break this down step by step."
    responses = [
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast and using the rest for muffins. Then, I can find out how much money she makes selling the remaining eggs at the market. Let's break this down step by step. \n\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\n```\nExecution Status: Done\nOutput: remaining_eggs = -4933815\n\nThought 2: The number of remaining eggs is negative, which doesn't make sense. I need to adjust the calculation to correctly determine the number of eggs remaining for sale.\n\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nmoney_made_per_day = remaining_eggs * 2\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nmoney_made_per_day = remaining_eggs * 2\n```\nExecution Status: Done\nOutput: money_made_per_day = 26\n\nThought 3: Janet makes $26 every day selling fresh duck eggs at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```"
    ]
    strategy = ReActMathStrategy(llm=FakeListChatModel(responses=responses))
    out = strategy.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert out == gt_out
    assert strategy._current_answer == ""
    assert strategy._scratchpad == gt_scratchpad
    assert not strategy._finished


def test_generate_action() -> None:
    """Tests ReActMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    
    gt_scratchpad = '\nAction: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]'
    gt_query = 'eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day'
    responses = [
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_used = eggs_for_breakfast + eggs_for_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]'
    ]
    strategy = ReActMathStrategy(llm=FakeListChatModel(responses=responses))
    action_type, query = strategy.generate_action(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        prompt=REACT_INSTRUCTION_GSM8K,
        additional_keys={}
    )
    assert action_type == "Calculate"
    assert query == gt_query
    assert strategy._current_answer == ""
    assert strategy._scratchpad == gt_scratchpad


def test_generate_observation() -> None:
    """Tests ReActMathStrategy generate_observation."""


def test_create_output_dict() -> None:
    """Tests ReActMathStrategy create_output_dict."""


def test_halting_condition() -> None:
    """Tests ReActMathStrategy halting_condition."""


def test_reset() -> None:
    """Tests ReActMathStrategy reset."""


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReActGSM8KStrategy(llm=llm)
    svamp_strategy = ReActSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReActTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReActGSM8KStrategy)
    assert isinstance(svamp_strategy, ReActSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReActTabMWPStrategy)