"""Unit tests for Reflexion Math strategies."""

import tiktoken

from agential.cog.fewshots.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.reflexion.output import (
    ReflexionCoTOutput,
    ReflexionCoTStepOutput,
    ReflexionReActOutput,
    ReflexionReActReActStepOutput,
    ReflexionReActStepOutput,
)
from agential.cog.reflexion.prompts import (
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
)
from agential.cog.reflexion.reflect import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.reflexion.strategies.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTMathStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActMathStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
)
from agential.llm.llm import BaseLLM, MockLLM
from agential.utils.metrics import PromptMetrics


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTMathStrategy init."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    gt_out = ReflexionCoTOutput(
        answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nearnings_per_egg = 2\ntotal_earnings = eggs_sold * earnings_per_egg\nanswer = total_earnings",
        total_prompt_tokens=20,
        total_completion_tokens=40,
        total_tokens=60,
        total_prompt_cost=3e-05,
        total_completion_cost=7.999999999999999e-05,
        total_cost=0.00010999999999999999,
        total_prompt_time=1.0,
        total_time=0.5,
        additional_info=[
            ReflexionCoTStepOutput(
                thought="Janet's ducks lay 16 eggs per day. Subtract the eggs eaten for breakfast from the total, then subtract the eggs used to make muffins. Finally, calculate the earnings by selling the remaining eggs at $2 per egg.Answer:",
                action_type="Finish",
                observation="Answer is CORRECT",
                answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nearnings_per_egg = 2\ntotal_earnings = eggs_sold * earnings_per_egg\nanswer = total_earnings",
                is_correct=True,
                reflections=[],
                thought_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                reflection_metrics=None,
            )
        ],
    )
    responses = [
        "Janet's ducks lay 16 eggs per day. Subtract the eggs eaten for breakfast from the total, then subtract the eggs used to make muffins. Finally, calculate the earnings by selling the remaining eggs at $2 per egg.\nAnswer: ",
        "Finish[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nearnings_per_egg = 2\ntotal_earnings = eggs_sold * earnings_per_egg\nanswer = total_earnings\n``` \n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTMathStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        reflect_prompt=REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=3,
        reset=True,
    )
    assert out == gt_out


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    responses = [
        "Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionCoTMathStrategy(llm=llm)
    scratchpad, action_type, query, action_metrics = strategy.generate_action(
        scratchpad="",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert (
        query
        == "eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day"
    )
    assert (
        scratchpad
        == "\nAction:  Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day\n```\n]"
    )
    assert action_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTMathStrategy generate_observation."""
    # Case 1: action_type is "Finish" and answer is correct.
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Finish",
        query="correct_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in scratchpad
    assert answer == "correct_answer"

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTMathStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Finish",
        query="incorrect_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in scratchpad
    assert answer == "incorrect_answer"

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTMathStrategy(llm=llm)
    scratchpad, answer, is_correct, obs = strategy.generate_observation(
        scratchpad="",
        action_type="Calculate",
        query="some_query",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Invalid action type, please try again."
    assert "Observation: Invalid action type, please try again." in scratchpad
    assert answer == ""


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTMathStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm, max_trials=3)

    assert strategy.halting_condition(3, "correct_answer", "correct_answer") == True

    assert strategy.halting_condition(2, "correct_answer", "correct_answer") == False

    assert strategy.halting_condition(2, "correct_answer", "correct_answer") == False


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTMathStrategy reflect_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionCoTMathStrategy(llm)

    assert not strategy.reflect_condition(0, "strategy1", "key1", "key2")
    assert strategy.reflect_condition(1, "strategy1", "key1", "key2")
    assert strategy.reflect_condition(1, "strategy1", "key2", "key2")
    assert strategy.reflect_condition(1, "", "key2", "key2")


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTMathStrategy instantiate strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionCoTSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionCoTTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionCoTGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionCoTSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionCoTTabMWPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActMathStrategy init."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActMathStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseLLM)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 3
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 5000
    assert isinstance(strategy.enc, tiktoken.Encoding)


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    gt_out = ReflexionReActOutput(
        answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
        total_prompt_tokens=180,
        total_completion_tokens=360,
        total_tokens=540,
        total_prompt_cost=0.00027,
        total_completion_cost=0.0007199999999999999,
        total_cost=0.00099,
        total_prompt_time=9.0,
        total_time=0.5,
        additional_info=[
            ReflexionReActStepOutput(
                steps=[
                    ReflexionReActReActStepOutput(
                        thought="Let's calculate how many eggs she has left after eating and baking for her friends.Thought 2: She lays 16 eggs per day, eats 3, bakes muffins with the rest, which leaves 16 - 3 = 13 eggs to sell.",
                        action_type="",
                        query="",
                        observation="Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
                        answer="",
                        external_tool_info={"execution_status": "", "code_answer": ""},
                        is_correct=False,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="Janet has 13 eggs left to sell at the farmers' market.",
                        action_type="Finish",
                        query="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_for_friends = 4933828\nremaining_eggs_to_sell = eggs_laid_per_day - eggs_eaten_for_breakfast\nsales_per_day = remaining_eggs_to_sell * 2\nanswer = sales_per_day",
                        observation="Answer is INCORRECT",
                        answer="eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_for_friends = 4933828\nremaining_eggs_to_sell = eggs_laid_per_day - eggs_eaten_for_breakfast\nsales_per_day = remaining_eggs_to_sell * 2\nanswer = sales_per_day",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 26,
                        },
                        is_correct=False,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                ],
                reflections=[],
                reflection_metrics=None,
            ),
            ReflexionReActStepOutput(
                steps=[
                    ReflexionReActReActStepOutput(
                        thought="Let's break down the relevant components. Janet's ducks lay 16 eggs per day, of which she eats 3 for breakfast. This means she has 16 - 3 = 13 eggs left to sell. Since she sells each fresh duck egg for $2, the total amount she makes daily can be calculated by multiplying the number of eggs she sells by the price per egg.Thought 2: To calculate how much in dollars Janet makes every day at the farmers' market, I need to multiply the number of eggs she sells daily (13) by the price per egg ($2).",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n```\nExecution Status: Done\nOutput: answer = 26",
                        answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 26,
                        },
                        is_correct=False,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="This plan is correct, and by following these steps, the answer should be accurate.",
                        action_type="",
                        query="",
                        observation="Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
                        answer="",
                        external_tool_info={"execution_status": "", "code_answer": ""},
                        is_correct=False,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I have successfully calculated that Janet makes $26 every day at the farmers' market by selling the remaining eggs from her ducks.",
                        action_type="Finish",
                        query="total_daily_income = 26\nanswer = total_daily_income",
                        observation="Answer is INCORRECT",
                        answer="total_daily_income = 26\nanswer = total_daily_income",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 26,
                        },
                        is_correct=False,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                ],
                reflections=[
                    "My reasoning failed because I mistakenly included the number of eggs she bakes for her friends (4933828) in the calculation, which skewed the result. In the future, I should carefully exclude irrelevant information and focus on the relevant calculations to accurately determine the amount she makes from selling eggs at the farmers' market. High-level Plan: Next time, I will correctly identify the relevant components needed for the calculation, such as the number of eggs she eats for breakfast and the price she sells the remaining eggs for. By accurately isolating these values and performing the correct arithmetic operations, I will calculate the total amount she makes daily at the farmers' market."
                ],
                reflection_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            ReflexionReActStepOutput(
                steps=[
                    ReflexionReActReActStepOutput(
                        thought="Let's accurately identify the relevant components for this calculation: the number of eggs laid per day, the eggs used for breakfast, the eggs used for muffins, and the price per fresh duck egg. We need to subtract the eggs used for breakfast and muffins to determine the eggs available for sale.Thought 2: To avoid the previous mistake, I should focus on the key figures: the number of eggs used for breakfast and the number of eggs baked into muffins. By correctly deducting these from the total eggs laid, I can determine the amount Janet has available to sell at the farmers' market.Plan: Calculate the total income by multiplying the number of eggs available for sale by the price per egg. Be careful not to include irrelevant information like the eggs used for muffins as part of the income calculation.",
                        action_type="",
                        query="",
                        observation="Invalid Action. Valid Actions are Calculate[code] and Finish[answer].",
                        answer="",
                        external_tool_info={"execution_status": "", "code_answer": ""},
                        is_correct=False,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I should focus on correctly identifying the relevant components, such as the number of eggs available for sale and the price per egg, to calculate Janet's daily income accurately. Plan: Calculate the total income by multiplying the number of eggs available for sale by the price per egg. Be careful not to include irrelevant information like the eggs used for muffins as part of the income calculation.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n```\nExecution Status: Done\nOutput: answer = -9867630",
                        answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": -9867630,
                        },
                        is_correct=True,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                    ReflexionReActReActStepOutput(
                        thought="I made a mistake in the calculation. The negative value for total_daily_income indicates an error in the subtraction of eggs_for_muffins. Let's correct it by ensuring the deduction is accurate and recalculating the total daily income correctly.",
                        action_type="Finish",
                        query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
                        observation="Answer is CORRECT",
                        answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": -9867630,
                        },
                        is_correct=True,
                        thought_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptMetrics(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                    ),
                ],
                reflections=[
                    "My reasoning failed because I mistakenly included the number of eggs she bakes for her friends (4933828) in the calculation, which skewed the result. In the future, I should carefully exclude irrelevant information and focus on the relevant calculations to accurately determine the amount she makes from selling eggs at the farmers' market. High-level Plan: Next time, I will correctly identify the relevant components needed for the calculation, such as the number of eggs she eats for breakfast and the price she sells the remaining eggs for. By accurately isolating these values and performing the correct arithmetic operations, I will calculate the total amount she makes daily at the farmers' market.",
                    "My reasoning failed because in the initial calculation, I forgot to account for the number of muffins Janet bakes for her friends each day (4933828) and mistakenly assumed that this number represented eggs. In the future, I should ensure that all quantities are correctly identified and included in the calculations. To mitigate this mistake, I will revise the plan by correctly reflecting the actual number of eggs Janet has to sell daily and include the quantity baked into muffins for her friends.Revised Plan:Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?Thought: Let's break down the relevant components. Since Janet eats 3 eggs for breakfast and bakes the remaining ones into muffins, we need to consider those additional deductions before determining the actual number of eggs she has to sell. By ensuring all components are accounted for accurately, the revised calculation will provide the correct answer for the daily income from selling fresh duck eggs at the farmers' market.Action: Calculate[```pythoneggs_laid_per_day = 16eggs_for_breakfast = 3eggs_for_muffins = 4933828eggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffinsprice_per_egg = 2total_daily_income = eggs_to_sell * price_per_egganswer = total_daily_income```]",
                ],
                reflection_metrics=PromptMetrics(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
        ],
    )
    responses = [
        "Let's calculate how many eggs she has left after eating and baking for her friends.\nThought 2: She lays 16 eggs per day, eats 3, bakes muffins with the rest, which leaves 16 - 3 = 13 eggs to sell.\nAction: Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nmoney_per_fresh_duck_egg = 2\nmoney_made_per_day = eggs_to_sell * money_per_fresh_duck_egg\nanswer = money_made_per_day\n```\n]",
        "```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nAction 2:",
        "Janet has 13 eggs left to sell at the farmers' market.\nAction 2: Finish[\n```python\neggs_left_to_sell = 13\neggs_sold_price = 2\nmoney_made_per_day = eggs_left_to_sell * eggs_sold_price\nanswer = money_made_per_day\n``` \n]",
        "Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_for_friends = 4933828\nremaining_eggs_to_sell = eggs_laid_per_day - eggs_eaten_for_breakfast\nsales_per_day = remaining_eggs_to_sell * 2\nanswer = sales_per_day\n``` \n]",
        "My reasoning failed because I mistakenly included the number of eggs she bakes for her friends (4933828) in the calculation, which skewed the result. In the future, I should carefully exclude irrelevant information and focus on the relevant calculations to accurately determine the amount she makes from selling eggs at the farmers' market. \n\nHigh-level Plan: Next time, I will correctly identify the relevant components needed for the calculation, such as the number of eggs she eats for breakfast and the price she sells the remaining eggs for. By accurately isolating these values and performing the correct arithmetic operations, I will calculate the total amount she makes daily at the farmers' market.",
        "Let's break down the relevant components. Janet's ducks lay 16 eggs per day, of which she eats 3 for breakfast. This means she has 16 - 3 = 13 eggs left to sell. Since she sells each fresh duck egg for $2, the total amount she makes daily can be calculated by multiplying the number of eggs she sells by the price per egg.\nThought 2: To calculate how much in dollars Janet makes every day at the farmers' market, I need to multiply the number of eggs she sells daily (13) by the price per egg ($2).\nAction: Finish[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\ntotal_income_daily = eggs_to_sell * price_per_egg\nanswer = total_income_daily\n```\n]",
        "Let's calculate the total amount Janet makes daily at the farmers' market.\n\nAction 2: Finish[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n```\n]",
        "This plan is correct, and by following these steps, the answer should be accurate.",
        "Finish[26]",
        "I have successfully calculated that Janet makes $26 every day at the farmers' market by selling the remaining eggs from her ducks.",
        "Finish[\n```python\ntotal_daily_income = 26\nanswer = total_daily_income\n``` \n]",
        "My reasoning failed because in the initial calculation, I forgot to account for the number of muffins Janet bakes for her friends each day (4933828) and mistakenly assumed that this number represented eggs. In the future, I should ensure that all quantities are correctly identified and included in the calculations. To mitigate this mistake, I will revise the plan by correctly reflecting the actual number of eggs Janet has to sell daily and include the quantity baked into muffins for her friends.\n\nRevised Plan:\nQuestion: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nThought: Let's break down the relevant components. Since Janet eats 3 eggs for breakfast and bakes the remaining ones into muffins, we need to consider those additional deductions before determining the actual number of eggs she has to sell. By ensuring all components are accounted for accurately, the revised calculation will provide the correct answer for the daily income from selling fresh duck eggs at the farmers' market.\nAction: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n```\n]",
        "Let's accurately identify the relevant components for this calculation: the number of eggs laid per day, the eggs used for breakfast, the eggs used for muffins, and the price per fresh duck egg. We need to subtract the eggs used for breakfast and muffins to determine the eggs available for sale.\nThought 2: To avoid the previous mistake, I should focus on the key figures: the number of eggs used for breakfast and the number of eggs baked into muffins. By correctly deducting these from the total eggs laid, I can determine the amount Janet has available to sell at the farmers' market.\nPlan: Calculate the total income by multiplying the number of eggs available for sale by the price per egg. Be careful not to include irrelevant information like the eggs used for muffins as part of the income calculation.\nAction: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\n\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\n\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n```\n]",
        "[```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\n```\n]",
        "I should focus on correctly identifying the relevant components, such as the number of eggs available for sale and the price per egg, to calculate Janet's daily income accurately. Plan: Calculate the total income by multiplying the number of eggs available for sale by the price per egg. Be careful not to include irrelevant information like the eggs used for muffins as part of the income calculation. Action: Calculate[```python```]",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n``` \n]",
        "I made a mistake in the calculation. The negative value for total_daily_income indicates an error in the subtraction of eggs_for_muffins. Let's correct it by ensuring the deduction is accurate and recalculating the total daily income correctly.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n```\n]",
        "Finish[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\neggs_to_sell = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ntotal_daily_income = eggs_to_sell * price_per_egg\nanswer = total_daily_income\n``` \n]",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActMathStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        reflect_prompt=REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=3,
        reset=True,
    )
    assert out == gt_out


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_scratchpad = "\nAction 0: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\n]"
    responses = [
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\n]"
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActMathStrategy(llm=llm)
    scratchpad, action_type, query, thought_metrics = strategy.generate_action(
        idx=0,
        scratchpad="",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert action_type == "Calculate"
    assert (
        query
        == "eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income"
    )
    assert scratchpad == gt_scratchpad
    assert thought_metrics == PromptMetrics(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActMathStrategy generate_observation."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActMathStrategy(llm=llm)

    # Test Calculate.
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=0,
            scratchpad="",
            action_type="Calculate",
            query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income",
            key=-9867630,
        )
    )
    assert is_correct
    assert (
        obs
        == "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\nExecution Status: Done\nOutput: answer = -9867630"
    )
    assert (
        scratchpad
        == "\nObservation 0: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\nExecution Status: Done\nOutput: answer = -9867630"
    )
    assert (
        answer
        == "eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income"
    )
    assert not finished
    assert external_tool_info == {"execution_status": "Done", "code_answer": -9867630}

    # Test Finish incorrect.
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=0,
            scratchpad="",
            action_type="Finish",
            query="answer = 5",
            key="key1",
        )
    )
    assert not is_correct
    assert obs == "Answer is INCORRECT"

    assert scratchpad == "\nObservation 0: Answer is INCORRECT"
    assert answer == "answer = 5"
    assert finished
    assert external_tool_info == {"execution_status": "Done", "code_answer": 5}

    # Test Finish correct.
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=0,
            scratchpad="",
            action_type="Finish",
            query="answer = 5",
            key=5,
        )
    )
    assert is_correct
    assert obs == "Answer is CORRECT"
    assert scratchpad == "\nObservation 0: Answer is CORRECT"
    assert answer == "answer = 5"
    assert finished
    assert external_tool_info == {"execution_status": "Done", "code_answer": 5}

    # Test invalid.
    scratchpad, answer, finished, is_correct, obs, external_tool_info = (
        strategy.generate_observation(
            idx=0,
            scratchpad="",
            action_type="Invalid",
            query="answer = 5",
            key=5,
        )
    )
    assert is_correct
    assert (
        obs == "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
    )
    assert (
        scratchpad
        == "\nObservation 0: Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
    )
    assert answer == ""
    assert not finished
    assert external_tool_info == {"execution_status": "", "code_answer": ""}


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActMathStrategy create_output_dict."""
    strategy = ReflexionReActMathStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))
    react_out = [
        {
            "thought": "First thought",
            "action_type": "Query",
            "query": "What is the capital of France?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
        }
    ]
    reflections = "Reflection on the first thought."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
        "prompt_metrics": {"reflection": None},
    }
    assert output == expected_output


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActMathStrategy react_create_output_dict."""
    strategy = ReflexionReActMathStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    # Test case 1: Valid output creation
    output = strategy.react_create_output_dict(
        thought="Initial thought",
        action_type="Query",
        query="What is the capital of France?",
        obs="Observation: Answer is CORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=True,
    )
    expected_output = {
        "thought": "Initial thought",
        "action_type": "Query",
        "query": "What is the capital of France?",
        "observation": "Observation: Answer is CORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": True,
        "prompt_metrics": {"thought": None, "action": None},
    }
    assert output == expected_output


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "correct_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=3)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer") == True

    # Test case 4: Halting condition met using max_trials from kwargs.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer", max_trials=4) == False

    # Test case 5: Halting condition not met using max_trials from kwargs.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer", max_trials=3) == True


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy react_halting_condition."""
    strategy = ReflexionReActMathStrategy(llm=MockLLM("gpt-3.5-turbo", responses=[]))

    idx = 0
    question = "What is the capital of France?"
    examples = ""
    reflections = ""
    prompt = "Answer the question."

    assert not strategy.react_halting_condition(
        idx, question, examples, reflections, prompt, {}
    )


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActMathStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = ReflexionReActMathStrategy(llm=llm)
    strategy._scratchpad = "Some previous state"
    strategy._finished = True

    strategy.reset()

    assert strategy._scratchpad == ""
    assert not strategy._finished
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}
    assert strategy._prompt_metrics == {"reflection": None}


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActMathStrategy reflect."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_reflections = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActMathStrategy(llm=llm)
    _, reflections = strategy.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert reflections == gt_reflections
    assert strategy._prompt_metrics_react == {"thought": None, "action": None}
    assert strategy._prompt_metrics == {
        "reflection": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_cost": 1.5e-05,
            "completion_tokens_cost": 3.9999999999999996e-05,
            "total_tokens_cost": 5.4999999999999995e-05,
            "time_sec": 0.5,
        }
    }


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActMathStrategy reflect_condition."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActMathStrategy(llm=llm)
    out = strategy.reflect_condition(
        step_idx=1,
        reflect_strategy="reflexion",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        key="key",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert not out


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActMathStrategy instantiate strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    gsm8k_strategy = ReflexionReActGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionReActSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionReActTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionReActGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionReActSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionReActTabMWPStrategy)
