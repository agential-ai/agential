"""Unit tests for LATS Math strategies."""

import itertools

from agential.cog.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.output import (
    LATSEvaluateMetrics,
    LATSGenerateMetrics,
    LATSReActStepOutput,
    LATSSimulationMetrics,
    LATSSimulationOutput,
    LATSSimulationStepMetrics,
    LATSStepOutput,
)
from agential.cog.lats.prompts import (
    GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_INSTRUCTION_GSM8K,
    LATS_REFLECT_INSTRUCTION_GSM8K,
    LATS_VALUE_INSTRUCTION_GSM8K,
)
from agential.cog.lats.strategies.math import (
    LATSGSM8KStrategy,
    LATSMathStrategy,
    LATSSVAMPStrategy,
    LATSTabMWPStrategy,
)
from agential.llm.llm import MockLLM
from agential.utils.metrics import Response


def test_init() -> None:
    """Test initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSMathStrategy(
        llm=llm,
        n_samples=5,
        max_reflections=4,
        depth_limit=7,
        max_unique=5,
        cache_values=True,
    )

    assert strategy.llm == llm
    assert strategy.n_samples == 5
    assert strategy.max_reflections == 4
    assert strategy.depth_limit == 7
    assert strategy.max_unique == 5
    assert strategy.cache_values is True
    assert strategy.root is None
    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}


def test_generate() -> None:
    """Test the generate method."""
    gt_terminal_node_state = {
        "state": LATSReActStepOutput(
            thought="The result is $9,867,630 that Janet makes every day at the farmers' market.",
            action_type="Finish",
            query="answer = 9867630",
            observation="Answer is INCORRECT",
            answer="answer = 9867630",
            external_tool_info={"execution_status": "Done", "code_answer": 9867630},
        ),
        "visits": 1,
        "value": -1.0,
        "depth": 5,
        "is_terminal": True,
        "reward": 0,
    }

    gt_additional_info = [
        LATSStepOutput(
            iteration=0,
            current_node={
                "state": LATSReActStepOutput(
                    thought="",
                    action_type="",
                    query="",
                    observation="",
                    answer="",
                    external_tool_info={},
                ),
                "visits": 0,
                "value": 0,
                "depth": 0,
                "is_terminal": False,
                "reward": 0,
            },
            children_nodes=[
                {
                    "state": LATSReActStepOutput(
                        thought="First, I need to calculate how many eggs Janet has left after eating three for breakfast and baking muffins.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None",
                        answer="",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": None,
                        },
                    ),
                    "visits": 0,
                    "value": 0.0,
                    "depth": 1,
                    "is_terminal": False,
                    "reward": 0,
                },
                {
                    "state": LATSReActStepOutput(
                        thought="I need to figure out how much money Janet makes at the farmers' market daily.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nearning_per_day = eggs_sold * 2\nanswer = earning_per_day",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nearning_per_day = eggs_sold * 2\nanswer = earning_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
                        answer="",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": -9867630,
                        },
                    ),
                    "visits": 0,
                    "value": 0.0,
                    "depth": 1,
                    "is_terminal": False,
                    "reward": 0,
                },
            ],
            generate_metrics=LATSGenerateMetrics(
                thoughts_metrics=[
                    Response(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    Response(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ],
                actions_metrics=[
                    Response(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    Response(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ],
                reflections_metrics=[],
            ),
            values=[
                {"explanation": "Explanation not found", "value": 0.0},
                {"explanation": "Explanation not found", "value": 0.0},
            ],
            evaluate_metrics=LATSEvaluateMetrics(
                values_metrics=[
                    Response(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                    Response(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                        prompt_cost=1.5e-05,
                        completion_cost=3.9999999999999996e-05,
                        total_cost=5.4999999999999995e-05,
                        prompt_time=0.5,
                    ),
                ]
            ),
            simulation_results=LATSSimulationOutput(
                simulation_reward=0.0,
                simulation_terminal_node={
                    "state": LATSReActStepOutput(
                        thought="The result is $9,867,630 that Janet makes every day at the farmers' market.",
                        action_type="Finish",
                        query="answer = 9867630",
                        observation="Answer is INCORRECT",
                        answer="answer = 9867630",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 9867630,
                        },
                    ),
                    "visits": 0,
                    "value": 0,
                    "depth": 5,
                    "is_terminal": True,
                    "reward": 0,
                },
                simulation_current_nodes=[
                    {
                        "state": LATSReActStepOutput(
                            thought="First, I need to calculate how many eggs Janet has left after eating three for breakfast and baking muffins.",
                            action_type="Calculate",
                            query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins",
                            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None",
                            answer="",
                            external_tool_info={
                                "execution_status": "Done",
                                "code_answer": None,
                            },
                        ),
                        "visits": 0,
                        "value": 0.0,
                        "depth": 1,
                        "is_terminal": False,
                        "reward": 0,
                    },
                    {
                        "state": LATSReActStepOutput(
                            thought="I need to calculate how much money Janet makes by selling the remaining eggs at the farmers' market.",
                            action_type="Calculate",
                            query="price_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily",
                            observation="\n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: NameError(\"name 'eggs_remaining' is not defined\")\nOutput: answer = None",
                            answer="",
                            external_tool_info={
                                "execution_status": "NameError(\"name 'eggs_remaining' is not defined\")",
                                "code_answer": None,
                            },
                        ),
                        "visits": 0,
                        "value": 0,
                        "depth": 2,
                        "is_terminal": False,
                        "reward": 0,
                    },
                    {
                        "state": LATSReActStepOutput(
                            thought="I forgot to define the variable `eggs_remaining` before using it in the calculation. I need to go back and correct that.",
                            action_type="Calculate",
                            query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily",
                            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = -9867630",
                            answer="",
                            external_tool_info={
                                "execution_status": "Done",
                                "code_answer": -9867630,
                            },
                        ),
                        "visits": 0,
                        "value": 0,
                        "depth": 3,
                        "is_terminal": False,
                        "reward": 0,
                    },
                    {
                        "state": LATSReActStepOutput(
                            thought="The calculation resulted in a negative value, which doesn't make sense. I need to review my calculations to find the error.",
                            action_type="Calculate",
                            query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result",
                            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\nExecution Status: Done\nOutput: answer = 9867630",
                            answer="",
                            external_tool_info={
                                "execution_status": "Done",
                                "code_answer": 9867630,
                            },
                        ),
                        "visits": 0,
                        "value": 0,
                        "depth": 4,
                        "is_terminal": False,
                        "reward": 0,
                    },
                ],
                simulation_children_nodes=[
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="I need to calculate how much money Janet makes by selling the remaining eggs at the farmers' market.",
                                action_type="Calculate",
                                query="price_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily",
                                observation="\n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: NameError(\"name 'eggs_remaining' is not defined\")\nOutput: answer = None",
                                answer="",
                                external_tool_info={
                                    "execution_status": "NameError(\"name 'eggs_remaining' is not defined\")",
                                    "code_answer": None,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 2,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="I also need to calculate how many eggs Janet sells at the farmers' market.",
                                action_type="Calculate",
                                query="eggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day",
                                observation="\n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\nExecution Status: NameError(\"name 'eggs_remaining' is not defined\")\nOutput: answer = None",
                                answer="",
                                external_tool_info={
                                    "execution_status": "NameError(\"name 'eggs_remaining' is not defined\")",
                                    "code_answer": None,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 2,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="I forgot to define the variable `eggs_remaining` before using it in the calculation. I need to go back and correct that.",
                                action_type="Calculate",
                                query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily",
                                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = -9867630",
                                answer="",
                                external_tool_info={
                                    "execution_status": "Done",
                                    "code_answer": -9867630,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 3,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="I forgot to define eggs_remaining before using it to calculate the money made daily. I need to go back and define eggs_remaining first.",
                                action_type="Calculate",
                                query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily",
                                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = -9867630",
                                answer="",
                                external_tool_info={
                                    "execution_status": "Done",
                                    "code_answer": -9867630,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 3,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="The calculation resulted in a negative value, which doesn't make sense. I need to review my calculations to find the error.",
                                action_type="Calculate",
                                query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result",
                                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\nExecution Status: Done\nOutput: answer = 9867630",
                                answer="",
                                external_tool_info={
                                    "execution_status": "Done",
                                    "code_answer": 9867630,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 4,
                            "is_terminal": False,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="The calculation seems to have resulted in a negative value, which doesn't make sense in this context. I need to review the subtraction logic.",
                                action_type="Calculate",
                                query="eggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nif eggs_remaining < 0:\n    eggs_remaining = 0\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily",
                                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nif eggs_remaining < 0:\n    eggs_remaining = 0\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = 0",
                                answer="",
                                external_tool_info={
                                    "execution_status": "Done",
                                    "code_answer": 0,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 4,
                            "is_terminal": False,
                            "reward": 0,
                        },
                    ],
                    [
                        {
                            "state": LATSReActStepOutput(
                                thought="The result is $9,867,630 that Janet makes every day at the farmers' market.",
                                action_type="Finish",
                                query="answer = 9867630",
                                observation="Answer is INCORRECT",
                                answer="answer = 9867630",
                                external_tool_info={
                                    "execution_status": "Done",
                                    "code_answer": 9867630,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 5,
                            "is_terminal": True,
                            "reward": 0,
                        },
                        {
                            "state": LATSReActStepOutput(
                                thought="Janet makes $9,867,630 every day at the farmers' market.",
                                action_type="Finish",
                                query="answer = 9867630",
                                observation="Answer is INCORRECT",
                                answer="answer = 9867630",
                                external_tool_info={
                                    "execution_status": "Done",
                                    "code_answer": 9867630,
                                },
                            ),
                            "visits": 0,
                            "value": 0,
                            "depth": 5,
                            "is_terminal": True,
                            "reward": 0,
                        },
                    ],
                ],
                simulation_values=[
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                    [
                        {"explanation": "Explanation not found", "value": 0.0},
                        {"explanation": "Explanation not found", "value": 0.0},
                    ],
                ],
            ),
            simulation_metrics=LATSSimulationMetrics(
                simulation_step_metrics=[
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(
                            values_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ]
                        ),
                    ),
                    LATSSimulationStepMetrics(
                        generate_metrics=LATSGenerateMetrics(
                            thoughts_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            actions_metrics=[
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                                Response(
                                    prompt_tokens=10,
                                    completion_tokens=20,
                                    total_tokens=30,
                                    prompt_cost=1.5e-05,
                                    completion_cost=3.9999999999999996e-05,
                                    total_cost=5.4999999999999995e-05,
                                    prompt_time=0.5,
                                ),
                            ],
                            reflections_metrics=[],
                        ),
                        evaluate_metrics=LATSEvaluateMetrics(values_metrics=[]),
                    ),
                ]
            ),
        )
    ]

    gt_failed_trajectories = [
        {
            "trajectory": "\nThought 1: First, I need to calculate how many eggs Janet has left after eating three for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None\nThought 2: I need to calculate how much money Janet makes by selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 2: \n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: NameError(\"name 'eggs_remaining' is not defined\")\nOutput: answer = None\nThought 3: I forgot to define the variable `eggs_remaining` before using it in the calculation. I need to go back and correct that.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = -9867630\nThought 4: The calculation resulted in a negative value, which doesn't make sense. I need to review my calculations to find the error.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\nExecution Status: Done\nOutput: answer = 9867630\nThought 5: The result is $9,867,630 that Janet makes every day at the farmers' market.\nAction 5: Finish[\n```python\nanswer = 9867630\n```\n]\nObservation 5: Answer is INCORRECT",
            "final_answer": "answer = 9867630",
        },
        {
            "trajectory": "\nThought 1: First, I need to calculate how many eggs Janet has left after eating three for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None\nThought 2: I need to calculate how much money Janet makes by selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 2: \n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: NameError(\"name 'eggs_remaining' is not defined\")\nOutput: answer = None\nThought 3: I forgot to define the variable `eggs_remaining` before using it in the calculation. I need to go back and correct that.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = -9867630\nThought 4: The calculation resulted in a negative value, which doesn't make sense. I need to review my calculations to find the error.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\nExecution Status: Done\nOutput: answer = 9867630\nThought 5: Janet makes $9,867,630 every day at the farmers' market.\nAction 5: Finish[\n```python\nanswer = 9867630\n```\n]\nObservation 5: Answer is INCORRECT",
            "final_answer": "answer = 9867630",
        },
    ]
    gt_value_cache = {
        "\nThought 1: First, I need to calculate how many eggs Janet has left after eating three for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None::": "The trajectory is incorrect because it fails to account for the fact that Janet cannot use more eggs for muffins than the total number of eggs laid per day. Therefore, the calculation of `eggs_remaining` is flawed, leading to a lack of answer. \n\nCorrectness score: 1",
        "\nThought 1: I need to figure out how much money Janet makes at the farmers' market daily.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nearning_per_day = eggs_sold * 2\nanswer = earning_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nearning_per_day = eggs_sold * 2\nanswer = earning_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630::": "The trajectory is incorrect because the calculation for `eggs_sold` subtracts the number of eggs eaten for breakfast and used for muffins from the total number of eggs laid per day, resulting in a negative value. This negative value then affects the calculation for `earning_per_day`, leading to an incorrect answer. The correct approach should be to calculate the total number of eggs sold at the farmers' market, not the total number of eggs remaining after breakfast and muffin making. \n\nCorrectness score: 2",
    }

    responses = [
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten - eggs_baked_into_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_eaten = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 2: The calculation resulted in a negative number because I subtracted the number of eggs baked into muffins incorrectly.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten = 3\neggs_baked_into_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten\neggs_sold = eggs_remaining - eggs_baked_into_muffins\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_eaten = 3\neggs_baked_into_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten\neggs_sold = eggs_remaining - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 3: The calculation is still incorrect. I need to properly account for the eggs baked into muffins.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten = 3\neggs_baked_into_muffins = 4933828\neggs_laid = eggs_laid_per_day\neggs_eaten_by_janet = eggs_eaten\neggs_used_for_muffins = eggs_baked_into_muffins\neggs_remaining = eggs_laid - eggs_eaten_by_janet - eggs_used_for_muffins\neggs_sold = eggs_remaining\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_eaten = 3\neggs_baked_into_muffins = 4933828\neggs_laid = eggs_laid_per_day\neggs_eaten_by_janet = eggs_eaten\neggs_used_for_muffins = eggs_baked_into_muffins\neggs_remaining = eggs_laid - eggs_eaten_by_janet - eggs_used_for_muffins\neggs_sold = eggs_remaining\n```\nExecution Status: Done\nOutput: eggs_sold = 13\nThought 4: Janet sells 13 fresh duck eggs daily at the farmers' market.\nAction 4: Finish[\n```python\nanswer = 13 * 2\n```\n]\nObservation 4: \n```python\nanswer = 13 * 2\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\n```\n]",
        "I need to figure out how much money Janet makes at the farmers' market daily.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\ntotal_income_per_day = eggs_sold * price_per_egg\nanswer = total_income_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\ntotal_income_per_day = eggs_sold * price_per_egg\nanswer = total_income_per_day\n```\nExecution Status: Done\nOutput: answer = 30\nThought 2: Janet makes $30 every day at the farmers' market.\nAction 2: Finish[\n```python\nanswer = 30\n```\n]\nObservation 2: \n```python\nanswer = 30\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nearning_per_day = eggs_sold * 2\nanswer = earning_per_day\n```\n]",
        "The trajectory is incorrect because it fails to account for the fact that Janet cannot use more eggs for muffins than the total number of eggs laid per day. Therefore, the calculation of `eggs_remaining` is flawed, leading to a lack of answer. \n\nCorrectness score: 1",
        "The trajectory is incorrect because the calculation for `eggs_sold` subtracts the number of eggs eaten for breakfast and used for muffins from the total number of eggs laid per day, resulting in a negative value. This negative value then affects the calculation for `earning_per_day`, leading to an incorrect answer. The correct approach should be to calculate the total number of eggs sold at the farmers' market, not the total number of eggs remaining after breakfast and muffin making. \n\nCorrectness score: 2",
        "I need to calculate how much money Janet makes by selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_daily = eggs_remaining * price_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_daily = eggs_remaining * price_per_egg\n```\nExecution Status: Done\nOutput: answer = -9867617",
        "Calculate[\n```python\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]",
        "I also need to calculate how many eggs Janet sells at the farmers' market.\nAction 2: Calculate[\n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_daily = eggs_sold_daily * price_per_egg\n```\n]\nObservation 2: \n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_daily = eggs_sold_daily * price_per_egg\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\n]\nObservation 2: \n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\nExecution Status: Done\nOutput: answer = -9867638\nThought 3: The negative value for income doesn't make sense. I need to review the calculations to correct the error.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\nExecution Status: Done\nOutput: answer = -19707642\nThought 4: The negative value for income still doesn't make sense. I need to reevaluate the calculations to find the error.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = income_per_day\n```\nExecution Status: Done\nOutput: answer = -19707760\nThought 5: There seems to be a recurring error. I should double-check my calculations.\nAction 5: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = abs(income_per_day)\n```\n]\nObservation 5: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\nincome_per_day = eggs_sold_daily * price_per_egg\nanswer = abs(income_per_day)\n```\nExecution Status: Done\nOutput: answer = 19707760\nThought 6: Janet makes $19,707,760 every day at the farmers' market.\nAction 6: Finish[\n```python\nanswer = 19707760\n```\n]\nObservation 6: \n```python\nanswer = 19707760\n```",
        "The trajectory is incorrect because the variable `eggs_remaining` was not defined or calculated before trying to use it to calculate the money made daily. The correct approach would be to ensure that the variable `eggs_remaining` is properly calculated before using it in subsequent calculations. \n\nCorrectness score: 3",
        "The trajectory is incorrect because the variable `eggs_remaining` was not defined before using it to calculate the income per day. The calculation for income per day is also incorrect as it does not consider the eggs eaten for breakfast and used for muffins. The correct approach would be to subtract the eggs eaten for breakfast and used for muffins from the eggs laid per day to find out how many eggs are available for sale, and then multiply this by the price per egg to calculate the income per day.\nCorrectness score: 2",
        "I forgot to define the variable `eggs_remaining` before using it in the calculation. I need to go back and correct that.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = 26\n\nThought 4: Janet makes $26 every day at the farmers' market.\nAction 4: Finish[\n```python\nanswer = 26\n```\n]\nObservation 4: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]",
        "I forgot to define eggs_remaining before using it to calculate the money made daily. I need to go back and define eggs_remaining first.\nAction 3: Calculate[\n```python\neggs_remaining = 16 - 3 - 4933828\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 3: \n```python\neggs_remaining = 16 - 3 - 4933828\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = -9867646",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]",
        "The trajectory is incorrect because it miscalculates the `eggs_remaining` variable, resulting in a negative value for the total money made daily at the farmers' market. The process of calculating the remaining eggs after breakfast and baking muffins is correct, but the mistake lies in using that value to compute the daily earnings. The error in the calculation led to an incorrect final answer. \nCorrectness score: 2",
        "The trajectory is incorrect because the calculation of eggs_remaining was done incorrectly, leading to a negative value and an inaccurate result for the money made daily. The correct approach would be to calculate eggs remaining by subtracting the eggs eaten for breakfast and used for muffins from the total eggs laid per day. The incorrect calculation of eggs_remaining caused the subsequent calculations to be wrong as well, resulting in an incorrect final answer. \n\nCorrectness score: 2",
        "The calculation resulted in a negative value, which doesn't make sense. I need to review my calculations to find the error.\nAction 4: Finish[\n```python\nanswer = -9867630\n```\n]\nObservation 4:\n```python\nanswer = -9867630\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = abs(money_made_daily)  # Taking the absolute value to ensure a positive result\n```\nExecution Status: Done\nOutput: answer = 9867630\nThought 5: Janet makes $9,867,630 every day at the farmers' market.\nAction 5: Finish[\n```python\nanswer = 9867630\n```\n]\nObservation 5: \n```python\nanswer = 9867630\n```",
        "The calculation seems to have resulted in a negative value, which doesn't make sense in this context. I need to review the subtraction logic.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nif eggs_remaining < 0:\n    eggs_remaining = 0\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nprice_per_egg = 2\nif eggs_remaining < 0:\n    eggs_remaining = 0\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = 32",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nif eggs_remaining < 0:\n    eggs_remaining = 0\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_eaten_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_eaten_daily - eggs_used_for_muffins\nif eggs_remaining < 0:\n    eggs_remaining = 0\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\nanswer = money_made_daily\n```\nExecution Status: Done\nOutput: answer = 0\nThought 5: Based on the corrected logic, Janet makes $0 every day at the farmers' market.\nAction 5: Finish[\n```python\nanswer = 0\n```\n]\nObservation 5: \n```python\nanswer = 0\n```",
        "The trajectory is incorrect because it miscalculated the number of eggs remaining after breakfast and baking muffins. This led to a negative result when calculating the amount of money made at the farmers' market. By correcting the calculation and taking the absolute value of the final result, the trajectory eventually arrived at the correct answer. The correctness score is 5, as the trajectory made mistakes in the initial calculations but corrected them eventually.",
        "The trajectory is incorrect because it miscalculated the number of eggs remaining after accounting for breakfast consumption and muffin production. The corrected approach fixed the negative value issue by setting the remaining eggs to zero if the calculated value is negative. However, the calculation logic still requires adjustment to accurately represent the scenario. The trajectory's final result of 0 is incorrect in the context of the problem. \n\nCorrectness score: 3",
        "The result is $9,867,630 that Janet makes every day at the farmers' market.",
        "Finish[\n```python\nanswer = 9867630\n```\n]\nObservation 5: \n```python\nanswer = 9867630\n```",
        "Janet makes $9,867,630 every day at the farmers' market. \nAction 5: Finish[\n```python\nanswer = 9867630\n```\n]\nObservation 5: \n```python\nanswer = 9867630\n```",
        "Finish[\n```python\nanswer = 9867630\n```\n]\nObservation 5: \n```python\nanswer = 9867630\n```",
    ]

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSMathStrategy(
        llm=llm,
        n_samples=2,
        max_reflections=4,
        depth_limit=5,
        max_unique=5,
        cache_values=True,
        testing=True,
    )
    out = strategy.generate(
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        value_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
        prompt=LATS_INSTRUCTION_GSM8K,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
        value_prompt=LATS_VALUE_INSTRUCTION_GSM8K,
        additional_keys={},
        reflect_additional_keys={},
        value_additional_keys={},
        max_iterations=1,
        reset=True,
    )
    assert out.answer.to_dict() == gt_terminal_node_state
    assert out.total_completion_cost == 0.00112
    assert out.total_completion_tokens == 560
    assert out.total_prompt_cost == 0.0004200000000000001
    assert out.total_prompt_tokens == 280
    assert out.total_tokens == 840
    assert out.total_cost == 0.0015400000000000001
    assert out.total_prompt_time == 14.0
    assert out.total_time == 0.5
    assert out.additional_info == gt_additional_info
    assert strategy.failed_trajectories == gt_failed_trajectories
    assert strategy.reflection_map == []
    assert strategy.value_cache == gt_value_cache
    assert strategy.root.to_dict() == {
        "state": LATSReActStepOutput(
            thought="",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        "visits": 1,
        "value": 0.0,
        "depth": 0,
        "is_terminal": False,
        "reward": 0,
    }


def test_generate_children_nodes() -> None:
    """Test the generate method."""
    gt_states = [
        LATSReActStepOutput(
            thought="I need to calculate how much money Janet makes daily at the farmers' market.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": -9867630},
        ),
        LATSReActStepOutput(
            thought="I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg",
            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\nExecution Status: Done\nOutput: answer = -9867630",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": -9867630},
        ),
        LATSReActStepOutput(
            thought="First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        LATSReActStepOutput(
            thought="First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast",
            observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        LATSReActStepOutput(
            thought="First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.",
            action_type="Calculate",
            query="eggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins",
            observation="\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
    ]

    responses = [
        "I need to calculate how much money Janet makes daily at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 2: The calculation resulted in a negative value, which doesn't make sense for money made. I need to review the calculation.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 9867650\nThought 3: Janet makes $9867650 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 9867650\n```\n]\nObservation 3:\n```python\nanswer = 9867650\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\n]",
        "I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867649\nThought 2: The calculation resulted in a negative number, which doesn't make sense in this context. I need to review the subtraction steps.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\n]",
        "First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: The negative result indicates an error in the calculations. I need to correct the computation to determine the correct number of eggs left for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933807\nThought 3: Another error occurred in the calculation. I need to correct the computation to determine the correct number of eggs available for sale.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = -9867614\nThought 4: The negative result indicates that there was an error in the calculation. I need to correct the computation to determine the accurate daily profit from selling duck eggs at the farmers' market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = 9867614\nThought 5: Janet makes $9867614 every day at the farmers' market.\nAction 5: Finish[\n```python\nprofit_per_day = 9867614\n```\n]\nObservation 5: \n```python\nprofit_per_day = 9867614\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\n]",
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\nExecution Status: Done\nOutput: eggs_remaining_per_day = 13\nThought 2: Now, I need to find out how much Janet earns from selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]",
        "First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 2: The calculation is incorrect because Janet cannot have negative eggs to sell. I need to review the subtraction.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = 5\nThought 3: Janet sells 5 fresh duck eggs every day at the farmers' market.\nAction 3: Finish[\n```python\neggs_sold = 5\n```\n]\nObservation 3:\n```python\neggs_sold = 5\n```",
        "Calculate[\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\n]",
    ]

    gt_generate_metrics = LATSGenerateMetrics(
        thoughts_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
        actions_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
        reflections_metrics=[],
    )

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSMathStrategy(llm=llm)

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    root = strategy.initialize()

    children_nodes, generate_metrics = strategy.generate_children_nodes(
        node=root,
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_GSM8K,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
        reflect_additional_keys={},
    )
    assert len(children_nodes) == 5

    for gt_state, node in zip(
        gt_states,
        children_nodes,
    ):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0

    assert generate_metrics == gt_generate_metrics

    # Test generate with reflections.
    gt_states = [
        LATSReActStepOutput(
            thought="To calculate how much money Janet makes at the farmers' market daily, I need to first find out how many eggs she has available for sale after consuming some and using some for baking.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_baking = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_baking",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_baking = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_baking\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        LATSReActStepOutput(
            thought="I need to calculate how many eggs Janet has available for sale at the farmers' market after consuming some for breakfast and baking muffins.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        LATSReActStepOutput(
            thought="First, I need to calculate how many eggs Janet has available for sale at the farmers' market each day.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
        LATSReActStepOutput(
            thought="I need to calculate the total earnings Janet makes by selling the remaining eggs at the farmers' market.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_muffins\nprice_per_egg = 2\ndaily_earnings = eggs_sold * price_per_egg\nanswer = daily_earnings",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_muffins\nprice_per_egg = 2\ndaily_earnings = eggs_sold * price_per_egg\nanswer = daily_earnings\n```\nExecution Status: Done\nOutput: answer = -9867630",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": -9867630},
        ),
        LATSReActStepOutput(
            thought="First, I need to find out how many eggs Janet has available for sale at the farmers' market.",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_for_friends = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_baked_for_friends",
            observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_for_friends = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_baked_for_friends\n```\nExecution Status: Done\nOutput: answer = None",
            answer="",
            external_tool_info={"execution_status": "Done", "code_answer": None},
        ),
    ]

    responses = [
        "My reasoning failed in this trial because I mistakenly used the total number of eggs laid per day instead of considering the number of eggs consumed by Janet. Going forward, I should pay closer attention to the specific requirements of the question and factor in all relevant information before calculating the final answer. To mitigate this failure, I will revise my plan to accurately calculate the number of eggs available for sale at the farmers' market based on the eggs laid and those consumed by Janet.",
        "My reasoning failed because I incorrectly focused on the wrong aspect of the question. Instead of calculating the total earnings from selling the duck eggs at the market, I mistakenly calculated the total number of eggs used for breakfast and baking muffins. In the future, I should carefully identify and prioritize the key components of the question before proceeding with calculations. The correct calculation should involve determining the number of eggs sold at the market daily and multiplying that by the price per egg to find the daily earnings.",
        "To calculate how much money Janet makes at the farmers' market daily, I need to first find out how many eggs she has available for sale after consuming some and using some for baking.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_baked_daily\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_baked_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 2: The calculation is incorrect because it resulted in a negative number of eggs available for sale. I need to revise my approach to accurately determine the number of eggs Janet can sell at the market.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\neggs_used_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - eggs_used_daily\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\neggs_used_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - eggs_used_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 3: I mistakenly used the total number of eggs consumed and baked daily instead of accounting for the number of eggs consumed in total. I need to adjust my calculation.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\ntotal_eggs_consumed_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - total_eggs_consumed_daily\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\ntotal_eggs_consumed_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - total_eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 4: My calculation still resulted in a negative number of eggs available for sale. I need to correct my approach to determine the correct number of eggs Janet can sell at the market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\ntotal_eggs_consumed_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - total_eggs_consumed_daily\n```\n]\nObservation 4:\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\ntotal_eggs_consumed_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - total_eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 5: My approach is still resulting in a negative number of eggs available for sale. I need to revisit my calculations and correct the error to determine Janet's daily earnings accurately.\nAction 5: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\ntotal_eggs_consumed_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - total_eggs_consumed_daily\n```\n]\nObservation 5:\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_daily = 4933828\ntotal_eggs_consumed_daily = eggs_consumed_daily + eggs_baked_daily\neggs_available_for_sale = eggs_laid_per_day - total_eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nFailed trajectory 1\n\nReflection: My reasoning failed in this trial because I mistakenly used the total number of eggs laid per day instead of considering the number of eggs consumed by Janet. Going forward, I should pay closer attention to the specific requirements of the question and factor in all relevant information before calculating the final answer. To mitigate this failure, I will revise my plan to accurately calculate the number of eggs available for sale at the farmers' market based on the eggs laid and those consumed by Janet.",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_baking = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_baking\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_baking = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_baking\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 2: My calculation yielded a negative number of eggs available for sale, which is not possible. I need to revisit and correct my calculation to determine the accurate number of eggs Janet can sell at the farmers' market.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_baking = 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_sold_daily * 2\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_baking = 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_sold_daily * 2\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "I need to calculate how many eggs Janet has available for sale at the farmers' market after consuming some for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828  # 3 for breakfast and the rest for muffins\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_available_for_sale * 2\nanswer = daily_earnings\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828  # 3 for breakfast and the rest for muffins\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_available_for_sale * 2\nanswer = daily_earnings\n```\nExecution Status: Done\nOutput: answer = -9866510\nThought 2: My calculation resulted in a negative value, indicating an error in my reasoning. I need to revise the calculation to accurately determine Janet's daily earnings at the farmers' market.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828  # 3 for breakfast and the rest for muffins\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_available_for_sale * 2\nanswer = daily_earnings  # Corrected calculation\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828  # 3 for breakfast and the rest for muffins\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_available_for_sale * 2\nanswer = daily_earnings  # Corrected calculation\n```\nExecution Status: Done\nOutput: answer = -9866510\nThought 3: The issue persists with the calculation. I need to reevaluate my approach to determine Janet's daily earnings accurately.\nAction 3: Finish[\n```python\nanswer = daily_earnings  # Corrected calculation\n```\n]\nObservation 3: \n```python\nanswer = -9866510\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933809\nThought 2: My calculation is incorrect because I mistakenly added the number of eggs consumed for breakfast and baking muffins instead of subtracting them. I need to revise the calculation to accurately determine the number of eggs available for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933809\nThought 3: I made the same mistake in my calculation. I need to subtract the total number of eggs consumed from the eggs laid to accurately determine the number of eggs available for sale.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933809\nThought 4: My calculation is still incorrect. I need to reevaluate my approach to accurately determine the number of eggs available for sale.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933809\nThought 5: My calculation still yields a negative value, indicating an error in my approach. I need to rethink my strategy to accurately determine the number of eggs available for sale.\nAction 5: Finish[\n```python\n# Please reevaluate the calculation to determine the number of eggs available for sale accurately.\n```\n]\nObservation 5: \n```python\n# Please reevaluate the calculation to determine the number of eggs available for sale accurately.\n```",
        "First, I need to calculate how many eggs Janet has available for sale at the farmers' market each day.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933815\nThought 2: My calculation resulted in a negative number of eggs for sale, indicating an error in my reasoning. I need to reassess the calculation and correct it to find the accurate number of eggs available for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933815\nThought 3: The negative value indicates a critical error in my calculation. I should reconsider my approach to determine the correct number of eggs available for sale after accounting for Janet's consumption and baking activities.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_baking = 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_for_baking\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_baking = 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_for_baking\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933815\nThought 4: My calculation still resulted in a negative value for the eggs available for sale. I need to revise my approach to accurately determine the number of eggs Janet can sell at the farmers' market each day.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_baking = 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_for_baking\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_for_baking = 4933828\neggs_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_for_baking\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933815\nThought 5: My attempts to correct the calculation have failed, resulting in the same negative value. I need to reassess the problem and identify the correct method to find the number of eggs Janet can sell each day at the farmers' market.\nAction 5: Finish[\n```python\n# No valid calculation found\n```\n]\nObservation 5: \n```python\n# No valid calculation found\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_remaining = 13\nThought 2: Now that I have the correct number of eggs available for sale, I need to calculate Janet's daily earnings at the farmers' market.\nAction 2: Calculate[\n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\ndaily_earnings = eggs_sold_daily * price_per_egg\n```\n]\nObservation 2: \n```python\neggs_sold_daily = eggs_remaining\nprice_per_egg = 2\ndaily_earnings = eggs_sold_daily * price_per_egg\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "I need to calculate the total earnings Janet makes by selling the remaining eggs at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_muffins\nearnings_daily = eggs_remaining * 2\nanswer = earnings_daily\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_muffins\nearnings_daily = eggs_remaining * 2\nanswer = earnings_daily\n```\nExecution Status: Done\nOutput: answer = -9867610\nThought 2: The negative earnings indicate an error in my calculation. I mistakenly used the eggs_used_for_muffins variable incorrectly. I should revise my approach to accurately calculate Janet's daily earnings at the farmers' market.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily\nearnings_daily = eggs_remaining * 2\nanswer = earnings_daily\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_daily\nearnings_daily = eggs_remaining * 2\nanswer = earnings_daily\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_muffins\nprice_per_egg = 2\ndaily_earnings = eggs_sold * price_per_egg\nanswer = daily_earnings\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed_daily - eggs_used_for_muffins\nprice_per_egg = 2\ndaily_earnings = eggs_sold * price_per_egg\nanswer = daily_earnings\n```\nExecution Status: Done\nOutput: answer = -9867632\nThought 2: The calculation resulted in a negative value, indicating an error in my reasoning. I must have made a mistake in determining the number of eggs sold at the market. I need to reconsider my approach to this problem.\nAction 2: Finish[\n```python\nanswer = -9867632\n```\n]\nObservation 2:\n```python\nanswer = -9867632\n```",
        "First, I need to find out how many eggs Janet has available for sale at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 2: My calculation is incorrect because I mistakenly added the number of eggs consumed for breakfast and muffins. I need to adjust my calculation to correctly determine the eggs available for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_sold_daily = -4933815\nThought 3: I need to correct my calculation by subtracting the total number of eggs consumed daily from the eggs laid per day to find out how many eggs Janet can sell at the farmers' market.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 4: My calculation is still incorrect. I need to rectify my mistake in calculating the eggs available for sale.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 5: My calculations are still incorrect. I need to reevaluate my approach to accurately determine the number of eggs available for sale.\nAction 5: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 5: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 6: My calculations are consistently incorrect. I need to rethink my approach and revise my strategy to accurately calculate the number of eggs available for sale.\nAction 6: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\n]\nObservation 6: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3 + 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 7: I have been unable to correct my calculation errors. I need to seek assistance or review my approach to accurately determine the number of eggs Janet can sell at the farmers' market.",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_for_friends = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_baked_for_friends\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_for_friends = 4933828\neggs_available_for_sale = eggs_laid_per_day - eggs_consumed_daily - eggs_baked_for_friends\n```\nExecution Status: Done\nOutput: eggs_available_for_sale = -4933815\nThought 2: The calculation is incorrect because it resulted in a negative number of eggs available for sale. I need to revise my approach and consider the question requirements more carefully.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_for_friends = 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_sold_daily * 2\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_daily = 3\neggs_baked_for_friends = 4933828\neggs_sold_daily = eggs_laid_per_day - eggs_consumed_daily\ndaily_earnings = eggs_sold_daily * 2\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSMathStrategy(llm=llm)

    strategy.failed_trajectories = [
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
        {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
        {
            "trajectory": "Failed trajectory 1",
            "final_answer": "Incorrect answer 1",
        },  # Duplicate, should be ignored
    ]

    root = strategy.initialize()
    children_nodes, generate_metrics = strategy.generate_children_nodes(
        node=root,
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_GSM8K,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
        reflect_additional_keys={},
    )
    gt_generate_metrics = LATSGenerateMetrics(
        thoughts_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
        actions_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
        reflections_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
    )

    assert len(children_nodes) == 5
    for gt_state, node in zip(
        gt_states,
        children_nodes,
    ):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0

    assert generate_metrics == gt_generate_metrics

    # Test case with a terminal child node (reward 0)
    gt_generate_metrics = LATSGenerateMetrics(
        thoughts_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            )
        ],
        actions_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            )
        ],
        reflections_metrics=[],
    )

    responses = [
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast and using some for muffins.\nAction 1: Calculate[\n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remain = eggs_per_day - eggs_used\n```\n]\nObservation 1: \n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remain = eggs_per_day - eggs_used\n```\nExecution Status: Done\nOutput: eggs_remain = -4933815\nThought 2: The calculation is incorrect because it resulted in a negative value for remaining eggs. I need to revise the calculation to account for the total eggs left.\nAction 2: Calculate[\n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remain = eggs_per_day - eggs_used\ntotal_income = eggs_remain * 2\n```\n]\nObservation 2: \n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remain = eggs_per_day - eggs_used\ntotal_income = eggs_remain * 2\n```\nExecution Status: Done\nOutput: total_income = -9867630\nThought 3: The calculation for total income is also incorrect because it resulted in a negative value. I need to correct the calculation to ensure the revenue is correctly calculated.\nAction 3: Calculate[\n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remain = eggs_per_day - eggs_used\nif eggs_remain > 0:\n    total_income = eggs_remain * 2\nelse:\n    total_income = 0\n```\n]\nObservation 3: \n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remain = eggs_per_day - eggs_used\nif eggs_remain > 0:\n    total_income = eggs_remain * 2\nelse:\n    total_income = 0\n```\nExecution Status: Done\nOutput: total_income = 2\nThought 4: Janet makes $2 every day at the farmers' market.\nAction 4: Finish[\n```python\nanswer = total_income\n```\n]\nObservation 4: \n```python\nanswer = 2\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remaining = eggs_laid_per_day - eggs_used\n```\n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSMathStrategy(llm=llm, n_samples=1)

    root = strategy.initialize()
    children_nodes, generate_metrics = strategy.generate_children_nodes(
        node=root,
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_GSM8K,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
        reflect_additional_keys={},
    )
    assert len(children_nodes) == 1
    assert (
        children_nodes[0].state.thought
        == "First, I need to calculate how many eggs Janet has left after eating three for breakfast and using some for muffins."
    )
    assert children_nodes[0].state.action_type == "Calculate"
    assert (
        children_nodes[0].state.query
        == "eggs_laid_per_day = 16\neggs_breakfast = 3\neggs_muffins = 4933828\neggs_used = eggs_breakfast + eggs_muffins\neggs_remaining = eggs_laid_per_day - eggs_used"
    )
    assert generate_metrics == gt_generate_metrics


def test_generate_action() -> None:
    """Test the generate_action method."""
    llm = MockLLM(
        "gpt-3.5-turbo", responses=["Calculate[```python\nresult = 2 + 2\n```]"]
    )
    strategy = LATSMathStrategy(llm=llm)

    question = "What is 2 + 2?"
    examples = "Example 1\nExample 2"
    trajectory = "Thought 1: I need to calculate 2 + 2."
    reflections = "Reflection 1\nReflection 2"
    depth = 0
    prompt = "Generate an action"
    additional_keys = {"key": "value"}

    trajectory, action_type, query, action_metrics = strategy.generate_action(
        question,
        examples,
        trajectory,
        reflections,
        depth,
        prompt,
        additional_keys,
    )
    assert (
        trajectory
        == "Thought 1: I need to calculate 2 + 2.\nAction 1:  Calculate[\n```python\nresult = 2 + 2\n```\n]"
    )
    assert action_type == "Calculate"
    assert query == "result = 2 + 2"

    assert action_metrics == Response(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_observation() -> None:
    """Test the generate_observation method."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = LATSMathStrategy(llm=llm)

    key = "4"
    trajectory = "Previous trajectory"

    # Test Finish action.
    finish_result = strategy.generate_observation(key, "Finish", "4", trajectory, 1)
    assert finish_result == (
        "Previous trajectory\nObservation 2: Answer is INCORRECT",
        0,
        "Answer is INCORRECT",
        True,
        {"execution_status": "Done", "code_answer": None},
    )

    # Test Calculate action.
    calculate_result = strategy.generate_observation(
        key, "Calculate", "result = 2 + 2", trajectory, 2
    )
    assert calculate_result == (
        "Previous trajectory\nObservation 3: \n```python\nresult = 2 + 2\n```\nExecution Status: Done\nOutput: answer = None",
        0,
        "\n```python\nresult = 2 + 2\n```\nExecution Status: Done\nOutput: answer = None",
        False,
        {"execution_status": "Done", "code_answer": None},
    )

    # Test invalid action.
    invalid_result = strategy.generate_observation(
        key, "Invalid", "query", trajectory, 3
    )
    assert invalid_result == (
        "Previous trajectory\nObservation 4: Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
        0,
        "Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
        False,
        {"execution_status": "", "code_answer": ""},
    )


def test_evaluate_node() -> None:
    """Test the evaluate_node method."""
    llm = MockLLM(
        "gpt-3.5-turbo",
        responses=["Explanation: Good trajectory. Correctness score: 8"],
    )
    strategy = LATSMathStrategy(llm=llm)

    root = strategy.initialize()
    child1 = Node(
        state=LATSReActStepOutput(
            thought="Child 1",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        parent=root,
    )
    child2 = Node(
        state=LATSReActStepOutput(
            thought="Child 2",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        parent=root,
        is_terminal=True,
    )

    root.children = [child1, child2]

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    prompt = "Evaluate this trajectory"

    strategy.reflection_map = [
        {
            "trajectory": "Failed trajectory",
            "reflection": "This trajectory failed because...",
        }
    ]

    values, values_evaluation_metrics = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )

    assert len(values) == 2
    assert values == [
        {"explanation": "Good trajectory.", "value": 0.8},
        {"explanation": "", "value": -10000000000.0},
    ]

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == [
        {
            "trajectory": "Failed trajectory",
            "reflection": "This trajectory failed because...",
        }
    ]
    assert strategy.value_cache == {
        "\nThought 1: Child 1::Question: What is the capital of France?\nFailed trajectory\n\nExplanation: This trajectory is incorrect as This trajectory failed because...\nCorrectness score: 1": "Explanation: Good trajectory. Correctness score: 8"
    }
    assert strategy.root == root

    assert child1.value == 0.8
    assert child2.value == 0  # Terminal node, value not updated.

    expected_value_metric = [
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        None,
    ]

    for i, value_met in zip(
        values_evaluation_metrics.values_metrics, expected_value_metric
    ):
        assert i == value_met

    # Test caching.
    strategy.cache_values = True
    cached_values, values_evaluation_metrics = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert cached_values == values
    assert values_evaluation_metrics.values_metrics == [None, None]

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == [
        {
            "trajectory": "Failed trajectory",
            "reflection": "This trajectory failed because...",
        }
    ]
    assert strategy.value_cache == {
        "\nThought 1: Child 1::Question: What is the capital of France?\nFailed trajectory\n\nExplanation: This trajectory is incorrect as This trajectory failed because...\nCorrectness score: 1": "Explanation: Good trajectory. Correctness score: 8"
    }
    assert strategy.root == root

    # Test with empty reflection_map.
    strategy.reflection_map = []
    empty_reflection_values, values_evaluation_metrics = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert values_evaluation_metrics.values_metrics == [
        Response(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            prompt_cost=1.5e-05,
            completion_cost=3.9999999999999996e-05,
            total_cost=5.4999999999999995e-05,
            prompt_time=0.5,
        ),
        None,
    ]

    assert empty_reflection_values == values

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {
        "\nThought 1: Child 1::Question: What is the capital of France?\nFailed trajectory\n\nExplanation: This trajectory is incorrect as This trajectory failed because...\nCorrectness score: 1": "Explanation: Good trajectory. Correctness score: 8",
        "\nThought 1: Child 1::": "Explanation: Good trajectory. Correctness score: 8",
    }
    assert strategy.root == root


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    responses = [
        "First, I need to calculate how many fresh duck eggs Janet has available to sell at the farmers' market daily.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs = eggs_laid_per_day - eggs_for_breakfast\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs = eggs_laid_per_day - eggs_for_breakfast\n```\nExecution Status: Done\nOutput: total_eggs = 13\nThought 2: I now need to calculate how much Janet makes daily from selling the fresh duck eggs.\nAction 2: Calculate[\n```python\nprice_per_egg = 2\nincome_daily = total_eggs * price_per_egg\n```\n]\nObservation 2: \n```python\nprice_per_egg = 2\nincome_daily = total_eggs * price_per_egg\n```\nExecution Status: Done\nOutput: income_daily = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\n```\n]",
        "I need to calculate how much money Janet makes every day at the farmers' market based on the eggs her ducks lay and her daily activities.\nAction 1: Calculate[\n```python\neggs_laid_daily = 16\neggs_eaten_daily = 3\neggs_used_for_muffins_daily = 4933828\neggs_remaining_daily = eggs_laid_daily - eggs_eaten_daily - eggs_used_for_muffins_daily\nincome_daily = eggs_remaining_daily * 2\nanswer = income_daily\n```\n]\nObservation 1:\n```python\neggs_laid_daily = 16\neggs_eaten_daily = 3\neggs_used_for_muffins_daily = 4933828\neggs_remaining_daily = eggs_laid_daily - eggs_eaten_daily - eggs_used_for_muffins_daily\nincome_daily = eggs_remaining_daily * 2\nanswer = income_daily\n```\nExecution Status: Done\nOutput: answer = -9873824\nThought 2: The calculation resulted in a negative value, which is not possible for income. I need to review the calculations and find the error.\nAction 2: Calculate[\n```python\neggs_laid_daily = 16\neggs_eaten_daily = 3\neggs_used_for_muffins_daily = 4933828\neggs_remaining_daily = eggs_laid_daily - eggs_eaten_daily - eggs_used_for_muffins_daily\ndaily_income = eggs_remaining_daily * 2\nanswer = abs(daily_income)\n```\n]\nObservation 2:\n```python\neggs_laid_daily = 16\neggs_eaten_daily = 3\neggs_used_for_muffins_daily = 4933828\neggs_remaining_daily = eggs_laid_daily - eggs_eaten_daily - eggs_used_for_muffins_daily\ndaily_income = eggs_remaining_daily * 2\nanswer = abs(daily_income)\n```\nExecution Status: Done\nOutput: answer = 9873824\nThought 3: On a daily basis, Janet can earn $9873824 at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 9873824\n```\n]\nObservation 3:\n```python\nanswer = 9873824\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_baking_muffins = 4933828\negg_price_at_market = 2\ntotal_eggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_baking_muffins\nmoney_made_per_day = total_eggs_sold * egg_price_at_market\nanswer = money_made_per_day\n```\n]",
        "The trajectory fails because it stops before completing the calculation for the total eggs available to sell. The action only subtracts the eggs used for breakfast from the eggs laid per day but does not take into account baking muffins and selling the remainder. Therefore, it does not capture the total number of eggs available for sale accurately.\n\nCorrectness score: 2",
        "The trajectory is incorrect because the calculation for `total_eggs_sold` is incorrect. It subtracts the eggs eaten for breakfast and used for baking muffins from the eggs laid per day, which results in a negative value for the total eggs sold. The calculation should involve subtracting the eggs used for breakfast and baking from the total eggs laid, and then using the remaining eggs to calculate the total money made per day. \nCorrectness score: 2\n\nOverall, the trajectory attempted to follow the correct steps of calculating the money made per day at the market but got stuck at the intermediate step due to incorrect calculation logic.",
        "The total eggs available for sale at the farmers' market are 13 fresh duck eggs per day.\nAction 2: Finish[\n```python\ntotal_eggs_available = 13\n```\n]\nObservation 2: \n```python\ntotal_eggs_available = 13\n```",
        "Finish[\n```python\ntotal_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily\n```\n]\nObservation 2: \n```python\nanswer = 26\n```",
        "The calculated value was not saved in the `answer` variable. I need to save the total eggs available to sell. \nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\nanswer = total_eggs_available\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\nanswer = total_eggs_available\n```\nExecution Status: Done\nOutput: answer = 13\nThought 3: I need to calculate the total earnings Janet makes daily by selling fresh duck eggs at the market.\nAction 3: Calculate[\n```python\neggs_available = 13\nprice_per_egg = 2\nearnings_daily = eggs_available * price_per_egg\nanswer = earnings_daily\n```\n]\nObservation 3: \n```python\neggs_available = 13\nprice_per_egg = 2\nearnings_daily = eggs_available * price_per_egg\nanswer = earnings_daily\n```\nExecution Status: Done\nOutput: answer = 26\nThought 4: Janet makes $26 daily by selling fresh duck eggs at the farmers' market.\nAction 4: Finish[\n```python\nanswer = 26\n```\n]\nObservation 4: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\nanswer = total_eggs_available\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\nanswer = total_eggs_available\n```\nExecution Status: Done\nOutput: answer = 13\nThought 3: Janet has 13 fresh duck eggs available to sell daily at the farmers' market.\nAction 3: Calculate[\n```python\neggs_for_friends = 4933828\nremaining_eggs = total_eggs_available - eggs_for_friends\nanswer = remaining_eggs * 2\n```\n]\nObservation 3: \n```python\neggs_for_friends = 4933828\nremaining_eggs = total_eggs_available - eggs_for_friends\nanswer = remaining_eggs * 2\n```\nExecution Status: Done\nOutput: answer = -9867642\nThought 4: The calculation resulted in a negative value because we subtracted the `eggs_for_friends` before subtracting it. I need to reorder the operations.\nAction 4: Calculate[\n```python\neggs_for_friends = 4933828\nremaining_eggs = total_eggs_available - eggs_for_friends\nfinal_eggs_available = total_eggs_available - eggs_for_friends\nanswer = final_eggs_available * 2\n```\n]\nObservation 4: \n```python\neggs_for_friends = 4933828\nremaining_eggs = total_eggs_available - eggs_for_friends\nfinal_eggs_available = total_eggs_available - eggs_for_friends\nanswer = final_eggs_available * 2\n```\nExecution Status: Done\nOutput: answer = -9867642\nThought 5: The value of `final_eggs_available` was incorrectly calculated. I need to update the calculation correctly.\nAction 5: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\neggs_for_friends = 4933828\nfinal_eggs_available = total_eggs_available - eggs_for_friends\nanswer = final_eggs_available * 2\n```\n]\nObservation 5:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\neggs_for_friends = 4933828\nfinal_eggs_available = total_eggs_available - eggs_for_friends\nanswer = final_eggs_available * 2\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 6: The calculation is incorrect. I should positively update the calculation by using absolute values for `eggs_for_friends`.\nAction 6: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\neggs_for_friends = 4933828\nfinal_eggs_available = abs(total_eggs_available - eggs_for_friends)\nanswer = final_eggs_available * 2\n```\n]\nObservation 6: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\neggs_for_friends = 4933828\nfinal_eggs_available = abs(total_eggs_available - eggs_for_friends)\nanswer = final_eggs_available * 2\n```\nExecution Status: Done\nOutput: answer = 12",
    ]
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630
    examples = GSM8K_FEWSHOT_EXAMPLES_REACT
    reflect_examples = GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT
    value_examples = GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE
    prompt = LATS_INSTRUCTION_GSM8K
    reflect_prompt = LATS_REFLECT_INSTRUCTION_GSM8K
    value_prompt = LATS_VALUE_INSTRUCTION_GSM8K
    additional_keys = {}
    reflect_additional_keys = {}
    value_additional_keys = {}

    strategy = LATSMathStrategy(
        llm=MockLLM("gpt-3.5-turbo", responses=responses), depth_limit=3, n_samples=2
    )
    root_node = strategy.initialize()

    (
        simulation_reward,
        simulation_terminal_node,
        simulation_current_nodes,
        simulation_children_nodes,
        simulation_values,
        simulation_metrics,
    ) = strategy.simulate_node(
        node=root_node,
        question=question,
        key=key,
        examples=examples,
        reflect_examples=reflect_examples,
        value_examples=value_examples,
        prompt=prompt,
        reflect_prompt=reflect_prompt,
        value_prompt=value_prompt,
        additional_keys=additional_keys,
        reflect_additional_keys=reflect_additional_keys,
        value_additional_keys=value_additional_keys,
    )

    assert simulation_reward == 0
    assert simulation_terminal_node.to_dict() == {
        "state": LATSReActStepOutput(
            thought="The total eggs available for sale at the farmers' market are 13 fresh duck eggs per day.",
            action_type="Finish",
            query="total_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily",
            observation="Answer is INCORRECT",
            answer="total_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily",
            external_tool_info={"execution_status": "Done", "code_answer": 26},
        ),
        "visits": 0,
        "value": 0,
        "depth": 2,
        "is_terminal": True,
        "reward": 0,
    }

    expected_current_nodes = [
        {
            "state": LATSReActStepOutput(
                thought="",
                action_type="",
                query="",
                observation="",
                answer="",
                external_tool_info={},
            ),
            "visits": 0,
            "value": 0,
            "depth": 0,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="First, I need to calculate how many fresh duck eggs Janet has available to sell at the farmers' market daily.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast",
                observation="\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None",
                answer="",
                external_tool_info={"execution_status": "Done", "code_answer": None},
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
    ]

    for expected_node, node in zip(expected_current_nodes, simulation_current_nodes):
        assert node.to_dict() == expected_node

    flattened_simulation_children_nodes = list(
        itertools.chain(*simulation_children_nodes)
    )

    expected_simulation_children_nodes = [
        {
            "state": LATSReActStepOutput(
                thought="First, I need to calculate how many fresh duck eggs Janet has available to sell at the farmers' market daily.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast",
                observation="\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None",
                answer="",
                external_tool_info={"execution_status": "Done", "code_answer": None},
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="I need to calculate how much money Janet makes every day at the farmers' market based on the eggs her ducks lay and her daily activities.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_baking_muffins = 4933828\negg_price_at_market = 2\ntotal_eggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_baking_muffins\nmoney_made_per_day = total_eggs_sold * egg_price_at_market\nanswer = money_made_per_day",
                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_baking_muffins = 4933828\negg_price_at_market = 2\ntotal_eggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_baking_muffins\nmoney_made_per_day = total_eggs_sold * egg_price_at_market\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
                answer="",
                external_tool_info={
                    "execution_status": "Done",
                    "code_answer": -9867630,
                },
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="The total eggs available for sale at the farmers' market are 13 fresh duck eggs per day.",
                action_type="Finish",
                query="total_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily",
                observation="Answer is INCORRECT",
                answer="total_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily",
                external_tool_info={"execution_status": "Done", "code_answer": 26},
            ),
            "visits": 0,
            "value": 0,
            "depth": 2,
            "is_terminal": True,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="The calculated value was not saved in the `answer` variable. I need to save the total eggs available to sell.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\nanswer = total_eggs_available",
                observation="\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\nanswer = total_eggs_available\n```\nExecution Status: Done\nOutput: answer = 13",
                answer="",
                external_tool_info={"execution_status": "Done", "code_answer": 13},
            ),
            "visits": 0,
            "value": 0,
            "depth": 2,
            "is_terminal": False,
            "reward": 0,
        },
    ]

    for expected_node, node in zip(
        expected_simulation_children_nodes, flattened_simulation_children_nodes
    ):
        assert node.to_dict() == expected_node

    assert simulation_values == [
        [
            {"explanation": "Explanation not found", "value": 0.0},
            {"explanation": "Explanation not found", "value": 0.0},
        ]
    ]

    gt_simulation_metrics = LATSSimulationMetrics(
        simulation_step_metrics=[
            LATSSimulationStepMetrics(
                generate_metrics=LATSGenerateMetrics(
                    thoughts_metrics=[
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ],
                    actions_metrics=[
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ],
                    reflections_metrics=[],
                ),
                evaluate_metrics=LATSEvaluateMetrics(
                    values_metrics=[
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ]
                ),
            ),
            LATSSimulationStepMetrics(
                generate_metrics=LATSGenerateMetrics(
                    thoughts_metrics=[
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ],
                    actions_metrics=[
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        Response(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ],
                    reflections_metrics=[],
                ),
                evaluate_metrics=LATSEvaluateMetrics(values_metrics=[]),
            ),
        ]
    )
    assert simulation_metrics == gt_simulation_metrics

    assert strategy.failed_trajectories == [
        {
            "trajectory": "\nThought 1: First, I need to calculate how many fresh duck eggs Janet has available to sell at the farmers' market daily.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\ntotal_eggs_available = eggs_laid_per_day - eggs_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None\nThought 2: The total eggs available for sale at the farmers' market are 13 fresh duck eggs per day.\nAction 2: Finish[\n```python\ntotal_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily\n```\n]\nObservation 2: Answer is INCORRECT",
            "final_answer": "total_eggs_available = 13\nprice_per_egg = 2\ntotal_money_daily = total_eggs_available * price_per_egg\nanswer = total_money_daily",
        }
    ]
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
    assert strategy.root.to_dict() == {
        "state": LATSReActStepOutput(
            thought="",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        "visits": 0,
        "value": 0,
        "depth": 0,
        "is_terminal": False,
        "reward": 0,
    }


def test_expand_node() -> None:
    """Test the expand_node method."""
    responses = [
        "I need to calculate how much money Janet makes daily at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 2: The calculation resulted in a negative value, which doesn't make sense for money made. I need to review the calculation.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 9867650\nThought 3: Janet makes $9867650 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 9867650\n```\n]\nObservation 3:\n```python\nanswer = 9867650\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\n]",
        "I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867649\nThought 2: The calculation resulted in a negative number, which doesn't make sense in this context. I need to review the subtraction steps.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\n]",
        "First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: The negative result indicates an error in the calculations. I need to correct the computation to determine the correct number of eggs left for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933807\nThought 3: Another error occurred in the calculation. I need to correct the computation to determine the correct number of eggs available for sale.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = -9867614\nThought 4: The negative result indicates that there was an error in the calculation. I need to correct the computation to determine the accurate daily profit from selling duck eggs at the farmers' market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = 9867614\nThought 5: Janet makes $9867614 every day at the farmers' market.\nAction 5: Finish[\n```python\nprofit_per_day = 9867614\n```\n]\nObservation 5: \n```python\nprofit_per_day = 9867614\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\n]",
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\nExecution Status: Done\nOutput: eggs_remaining_per_day = 13\nThought 2: Now, I need to find out how much Janet earns from selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]",
        "First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 2: The calculation is incorrect because Janet cannot have negative eggs to sell. I need to review the subtraction.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = 5\nThought 3: Janet sells 5 fresh duck eggs every day at the farmers' market.\nAction 3: Finish[\n```python\neggs_sold = 5\n```\n]\nObservation 3:\n```python\neggs_sold = 5\n```",
        "Calculate[\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = LATSMathStrategy(llm=llm)

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    root = strategy.initialize()

    children_nodes, generate_metrics = strategy.expand_node(
        node=root,
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
        prompt=LATS_INSTRUCTION_GSM8K,
        reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
        reflect_additional_keys={},
    )

    expected_node = [
        {
            "state": LATSReActStepOutput(
                thought="I need to calculate how much money Janet makes daily at the farmers' market.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day",
                observation="\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
                answer="",
                external_tool_info={
                    "execution_status": "Done",
                    "code_answer": -9867630,
                },
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg",
                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\nExecution Status: Done\nOutput: answer = -9867630",
                answer="",
                external_tool_info={
                    "execution_status": "Done",
                    "code_answer": -9867630,
                },
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins",
                observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: answer = None",
                answer="",
                external_tool_info={"execution_status": "Done", "code_answer": None},
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast",
                observation="\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None",
                answer="",
                external_tool_info={"execution_status": "Done", "code_answer": None},
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
        {
            "state": LATSReActStepOutput(
                thought="First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.",
                action_type="Calculate",
                query="eggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins",
                observation="\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\nExecution Status: Done\nOutput: answer = None",
                answer="",
                external_tool_info={"execution_status": "Done", "code_answer": None},
            ),
            "visits": 0,
            "value": 0,
            "depth": 1,
            "is_terminal": False,
            "reward": 0,
        },
    ]

    for expected_node, node in zip(expected_node, children_nodes):
        assert node.to_dict() == expected_node

    gt_generate_metrics = LATSGenerateMetrics(
        thoughts_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
        actions_metrics=[
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
            Response(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                prompt_cost=1.5e-05,
                completion_cost=3.9999999999999996e-05,
                total_cost=5.4999999999999995e-05,
                prompt_time=0.5,
            ),
        ],
        reflections_metrics=[],
    )
    assert generate_metrics == gt_generate_metrics

    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}
    assert strategy.root.to_dict() == {
        "state": LATSReActStepOutput(
            thought="",
            action_type="",
            query="",
            observation="",
            answer="",
            external_tool_info={},
        ),
        "visits": 0,
        "value": 0,
        "depth": 0,
        "is_terminal": False,
        "reward": 0,
    }


def test_instantiate_strategies() -> None:
    """Test the instantiation of various LATS Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = LATSGSM8KStrategy(llm=llm)
    svamp_strategy = LATSSVAMPStrategy(llm=llm)
    tabmwp_strategy = LATSTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, LATSGSM8KStrategy)
    assert isinstance(svamp_strategy, LATSSVAMPStrategy)
    assert isinstance(tabmwp_strategy, LATSTabMWPStrategy)
