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
from agential.utils.metrics import PromptInfo


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
                thought_metrics=PromptInfo(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptInfo(
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
    assert action_metrics == PromptInfo(
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
        answer="answer = 0",
        total_prompt_tokens=240,
        total_completion_tokens=480,
        total_tokens=720,
        total_prompt_cost=0.0003600000000000001,
        total_completion_cost=0.0009599999999999999,
        total_cost=0.00132,
        total_prompt_time=12.0,
        total_time=0.5,
        additional_info=[
            ReflexionReActStepOutput(
                steps=[
                    ReflexionReActReActStepOutput(
                        thought="First, I need to calculate the total number of eggs laid per day after Janet eats three for breakfast.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed\n```\nExecution Status: Done\nOutput: answer = None",
                        answer="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": None,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="I need to calculate how much money Janet makes from selling the remaining eggs at the farmers' market daily.",
                        action_type="Calculate",
                        query="eggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day",
                        observation="\n```python\neggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\nExecution Status: Done\nOutput: answer = 26",
                        answer="eggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 26,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="Janet makes $26 every day at the farmers' market.",
                        action_type="Finish",
                        query="answer = 26",
                        observation="Answer is INCORRECT",
                        answer="answer = 26",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 26,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="First, I need to calculate how many eggs are left after Janet eats breakfast and bakes muffins for her friends.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None",
                        answer="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": None,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="There seems to be an error in the calculation of eggs remaining. I need to adjust my approach to correctly calculate the number of eggs left for Janet to sell at the farmers' market.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = None",
                        answer="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": None,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="I have successfully calculated the number of eggs Janet has left for selling. Now, I need to determine how much money she makes from selling these eggs at the farmers' market.",
                        action_type="Calculate",
                        query="price_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day",
                        observation="\n```python\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: NameError(\"name 'eggs_remaining' is not defined\")\nOutput: answer = None",
                        answer="price_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day",
                        external_tool_info={
                            "execution_status": "NameError(\"name 'eggs_remaining' is not defined\")",
                            "code_answer": None,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="I need to ensure that the variable `eggs_remaining` is defined before using it to calculate the money made at the farmers' market. I will revise my plan to include this step explicitly. Let's try again.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
                        answer="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": -9867630,
                        },
                        is_correct=True,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="The calculation outcome is negative, which doesn't make sense as Janet is selling eggs at the market. I need to review my plan and make sure I'm correctly subtracting the consumed eggs and eggs used for muffins before calculating the money made at the farmers' market. Let's refine the approach.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630",
                        answer="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": -9867630,
                        },
                        is_correct=True,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="The outcome is still incorrect. I need to reevaluate my approach and ensure I am correctly calculating the remaining eggs for sale and the money made at the farmers' market. Let's revise the plan to accurately address the issue.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\nif eggs_remaining >= 0:\n    price_per_egg = 2\n    money_made_per_day = eggs_remaining * price_per_egg\nelse:\n    money_made_per_day = 0\nanswer = money_made_per_day",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\nif eggs_remaining >= 0:\n    price_per_egg = 2\n    money_made_per_day = eggs_remaining * price_per_egg\nelse:\n    money_made_per_day = 0\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 0",
                        answer="eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\nif eggs_remaining >= 0:\n    price_per_egg = 2\n    money_made_per_day = eggs_remaining * price_per_egg\nelse:\n    money_made_per_day = 0\nanswer = money_made_per_day",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 0,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                    "My reasoning failed because I overlooked the step where Janet bakes muffins for her friends daily before selling the remainder at the farmers' market. In the future, I should ensure to include this intermediate step in my calculations for accurate results. To mitigate this failure, I will revise my plan by accounting for the number of eggs used for baking muffins before calculating the money made at the farmers' market."
                ],
                reflection_metrics=PromptInfo(
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
                        thought="I need to calculate how many eggs Janet has remaining after taking out breakfast eggs and the ones used for baking muffins before selling the rest at the market.",
                        action_type="Calculate",
                        query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nmoney_made_per_day = 2 * remaining_eggs if remaining_eggs >= 0 else 0\nanswer = money_made_per_day",
                        observation="\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nmoney_made_per_day = 2 * remaining_eggs if remaining_eggs >= 0 else 0\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 0",
                        answer="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nmoney_made_per_day = 2 * remaining_eggs if remaining_eggs >= 0 else 0\nanswer = money_made_per_day",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 0,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                        thought="I correctly calculated that Janet makes zero dollars per day at the farmers' market after accounting for breakfast eggs and muffin baking eggs.",
                        action_type="Finish",
                        query="answer = 0",
                        observation="Answer is INCORRECT",
                        answer="answer = 0",
                        external_tool_info={
                            "execution_status": "Done",
                            "code_answer": 0,
                        },
                        is_correct=False,
                        thought_metrics=PromptInfo(
                            prompt_tokens=10,
                            completion_tokens=20,
                            total_tokens=30,
                            prompt_cost=1.5e-05,
                            completion_cost=3.9999999999999996e-05,
                            total_cost=5.4999999999999995e-05,
                            prompt_time=0.5,
                        ),
                        action_metrics=PromptInfo(
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
                    "My reasoning failed because I overlooked the step where Janet bakes muffins for her friends daily before selling the remainder at the farmers' market. In the future, I should ensure to include this intermediate step in my calculations for accurate results. To mitigate this failure, I will revise my plan by accounting for the number of eggs used for baking muffins before calculating the money made at the farmers' market.",
                    "My reasoning failed because I did not properly consider the negative outcome when calculating the remaining eggs. Janet cannot have negative eggs to sell. To mitigate this failure, I should explicitly check for negative values in the remaining eggs calculation and set the money made per day to zero in such cases. This ensures a more accurate representation of the scenario and prevents illogical results.",
                ],
                reflection_metrics=PromptInfo(
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
        "First, I need to calculate the total number of eggs laid per day after Janet eats three for breakfast.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: eggs_remaining = 13\nThought 2: I have determined the number of eggs remaining after breakfast. Now, I need to calculate how much Janet earns daily at the farmers' market with the remaining eggs.\nAction 2: Calculate[\n```python\neggs_remaining = 13\nprice_per_egg = 2\nearnings_per_day = eggs_remaining * price_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining = 13\nprice_per_egg = 2\nearnings_per_day = eggs_remaining * price_per_egg\n```\nExecution Status: Done\nOutput: earnings_per_day = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed\n```\n]",
        "I need to calculate how much money Janet makes from selling the remaining eggs at the farmers' market daily.\nAction 2: Calculate[\n```python\neggs_remaining = 13\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\n```\n]\nObservation 2: \n```python\neggs_remaining = 13\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\n]\nObservation 2: \n```python\neggs_remaining = 13\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day by selling the remaining duck eggs at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "My reasoning failed because I overlooked the step where Janet bakes muffins for her friends daily before selling the remainder at the farmers' market. In the future, I should ensure to include this intermediate step in my calculations for accurate results. To mitigate this failure, I will revise my plan by accounting for the number of eggs used for baking muffins before calculating the money made at the farmers' market.",
        "First, I need to calculate how many eggs are left after Janet eats breakfast and bakes muffins for her friends.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_breakfast - eggs_used_for_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_breakfast - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: My calculation for eggs remaining is incorrect. I mistakenly subtracted the number of eggs used for muffins instead of adding it. I need to modify the calculation to add the eggs used for muffins.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_breakfast + eggs_used_for_muffins\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_breakfast + eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: eggs_remaining = 4933821\nThought 3: Now that I have the correct number of eggs remaining, I can calculate how much money Janet makes at the farmers' market daily.\nAction 3: Finish[\n```python\nmoney_per_egg = 2\nmoney_made_per_day = eggs_remaining * money_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 3: \n```python\nanswer = 9867642\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: The calculation result is not logically correct as the number of eggs remaining cannot be negative. I need to revise the plan to accurately compute the eggs remaining.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_bought_at_market = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_sold_at_market = eggs_bought_at_market - eggs_used_for_muffins\ndollars_per_egg = 2\nmoney_made_per_day = eggs_sold_at_market * dollars_per_egg\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_bought_at_market = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_sold_at_market = eggs_bought_at_market - eggs_used_for_muffins\ndollars_per_egg = 2\nmoney_made_per_day = eggs_sold_at_market * dollars_per_egg\n```\nExecution Status: Done\nOutput: money_made_per_day = -4933860\nThought 3: The calculated negative amount of money made per day is incorrect. I should review the calculation steps to identify the error and adjust the calculation as needed.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_sold_at_market = eggs_remaining - eggs_used_for_muffins\ndollars_per_egg = 2\nmoney_made_per_day = max(eggs_sold_at_market, 0) * dollars_per_egg\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_sold_at_market = eggs_remaining - eggs_used_for_muffins\ndollars_per_egg = 2\nmoney_made_per_day = max(eggs_sold_at_market, 0) * dollars_per_egg\n```\nExecution Status: Done\nOutput: money_made_per_day = 26\nThought 4: The amount of money Janet makes at the farmers' market daily is $26.\nAction 4: Finish[\n```python\nanswer = 26\n```\n]\nObservation 4: \n```python\nanswer = 26\n```",
        "There seems to be an error in the calculation of eggs remaining. I need to adjust my approach to correctly calculate the number of eggs left for Janet to sell at the farmers' market. \nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = -4933815\nThought 3: The calculation result is negative which indicates an error in the logic. I need to revise my plan to accurately determine the number of eggs remaining for Janet to sell at the farmers' market. \nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = -4933815",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n```\nExecution Status: Done\nOutput: answer = -4933807\nThought 3: The answer is negative, indicating an error in the calculation of eggs remaining. I will correct this by ensuring to account for the muffins Janet bakes daily.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins_per_day = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins_per_day\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins_per_day = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins_per_day\n```\nExecution Status: Done\nOutput: answer = -4933827\nThought 4: The calculation is still not correct. I need to revise my approach to correctly calculate the number of eggs left for selling at the farmers' market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\ntotal_eggs_used = eggs_consumed_for_breakfast + eggs_used_for_muffins\neggs_remaining = eggs_laid_per_day - total_eggs_used\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\ntotal_eggs_used = eggs_consumed_for_breakfast + eggs_used_for_muffins\neggs_remaining = eggs_laid_per_day - total_eggs_used\n```\nExecution Status: Done\nOutput: answer = 10\nThought 5: Janet makes $2 for every fresh duck egg she sells. I need to calculate the total amount she makes daily at the farmers' market.\nAction 5: Calculate[\n```python\nprice_per_egg = 2\ndaily_earnings = price_per_egg * eggs_remaining\n```\n]\nObservation 5: \n```python\nprice_per_egg = 2\ndaily_earnings = price_per_egg * eggs_remaining\n```\nExecution Status: Done\nOutput: answer = 20\nThought 6: Janet makes $20 every day at the farmers' market. This calculation is now correct.\nAction 6: Finish[\n```python\nanswer = 20\n```\n]\nObservation 6: \n```python\nanswer = 20\n```",
        "I have successfully calculated the number of eggs Janet has left for selling. Now, I need to determine how much money she makes from selling these eggs at the farmers' market.\nAction 3: Calculate[\n```python\neggs_remaining = 13  # Updated value after adjustments\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\n```\n]\nObservation 3: \n```python\neggs_remaining = 13  # Updated value after adjustments\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\n```\nExecution Status: Done\nOutput: answer = 26\nThought 4: Janet makes $26 every day at the farmers' market.\nAction 4: Finish[\n```python\nanswer = 26\n```\n]\nObservation 4: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 3: \n```python\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867647",
        "I need to ensure that the variable `eggs_remaining` is defined before using it to calculate the money made at the farmers' market. I will revise my plan to include this step explicitly. Let's try again.\n\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 4: \n```python\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867646",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_remaining -= eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]",
        "The calculation outcome is negative, which doesn't make sense as Janet is selling eggs at the market. I need to review my plan and make sure I'm correctly subtracting the consumed eggs and eggs used for muffins before calculating the money made at the farmers' market. Let's refine the approach.\nAction 5: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 5: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 6: There seems to be a recurring error in the calculation process. I will review and ensure that all the deductions are correctly made before calculating the final answer. I will reevaluate the plan to ensure accurate results. Let's correct the calculation approach once more.\nAction 6: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 6: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\n\nprice_per_egg = 2\nmoney_made_per_day = eggs_remaining * price_per_egg\nanswer = money_made_per_day\n```\n]",
        "The outcome is still incorrect. I need to reevaluate my approach and ensure I am correctly calculating the remaining eggs for sale and the money made at the farmers' market. Let's revise the plan to accurately address the issue.",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\nif eggs_remaining >= 0:\n    price_per_egg = 2\n    money_made_per_day = eggs_remaining * price_per_egg\nelse:\n    money_made_per_day = 0\nanswer = money_made_per_day\n```\n]\nObservation 6: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_used_for_muffins\nif eggs_remaining >= 0:\n    price_per_egg = 2\n    money_made_per_day = eggs_remaining * price_per_egg\nelse:\n    money_made_per_day = 0\nanswer = money_made_per_day\n```",
        "My reasoning failed because I did not properly consider the negative outcome when calculating the remaining eggs. Janet cannot have negative eggs to sell. To mitigate this failure, I should explicitly check for negative values in the remaining eggs calculation and set the money made per day to zero in such cases. This ensures a more accurate representation of the scenario and prevents illogical results.",
        "I need to calculate how many eggs Janet has remaining after taking out breakfast eggs and the ones used for baking muffins before selling the rest at the market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nif remaining_eggs < 0:\n    remaining_eggs = 0\nmoney_per_day = remaining_eggs * 2\nanswer = money_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nif remaining_eggs < 0:\n    remaining_eggs = 0\nmoney_per_day = remaining_eggs * 2\nanswer = money_per_day\n```\nExecution Status: Done\nOutput: answer = 26",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nmoney_made_per_day = 2 * remaining_eggs if remaining_eggs >= 0 else 0\nanswer = money_made_per_day\n```\n]",
        "I correctly calculated that Janet makes zero dollars per day at the farmers' market after accounting for breakfast eggs and muffin baking eggs. \nAction 2: Finish[\n```python\nanswer = 0\n```\n]\nObservation 2: \n```python\nanswer = 0\n```",
        "Finish[\n```python\nanswer = 0\n```\n]\nObservation 2: \n```python\nanswer = 0\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActMathStrategy(llm=llm, testing=True)
    out = strategy.generate(
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        prompt=REFLEXION_REACT_INSTRUCTION_GSM8K,
        reflect_examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        reflect_prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        reflect_strategy="reflexion",
        additional_keys={},
        reflect_additional_keys={},
        patience=3,
        reset=True,
    )
    assert out == gt_out


def test_reflexion_react_generate_react() -> None:
    """Tests ReflexionReActMathStrategy generate_react."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    gt_out = (
        4,
        False,
        "\nThought 1: First, I need to calculate the total number of eggs laid per day after Janet eats three for breakfast.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed\n```\nExecution Status: Done\nOutput: answer = None\nThought 2: I need to calculate how much money Janet makes from selling the remaining eggs at the farmers' market daily.\nAction 2: Calculate[\n```python\neggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\n]\nObservation 2: \n```python\neggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: Answer is INCORRECT",
        True,
        "answer = 26",
        [
            ReflexionReActReActStepOutput(
                thought="First, I need to calculate the total number of eggs laid per day after Janet eats three for breakfast.",
                action_type="Calculate",
                query="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed",
                observation="\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed\n```\nExecution Status: Done\nOutput: answer = None",
                answer="eggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed",
                external_tool_info={"execution_status": "Done", "code_answer": None},
                is_correct=False,
                thought_metrics=PromptInfo(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptInfo(
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
                thought="I need to calculate how much money Janet makes from selling the remaining eggs at the farmers' market daily.",
                action_type="Calculate",
                query="eggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day",
                observation="\n```python\neggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\nExecution Status: Done\nOutput: answer = 26",
                answer="eggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day",
                external_tool_info={"execution_status": "Done", "code_answer": 26},
                is_correct=False,
                thought_metrics=PromptInfo(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptInfo(
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
                thought="Janet makes $26 every day at the farmers' market.",
                action_type="Finish",
                query="answer = 26",
                observation="Answer is INCORRECT",
                answer="answer = 26",
                external_tool_info={"execution_status": "Done", "code_answer": 26},
                is_correct=False,
                thought_metrics=PromptInfo(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                action_metrics=PromptInfo(
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
        "First, I need to calculate the total number of eggs laid per day after Janet eats three for breakfast.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: eggs_remaining = 13\nThought 2: I have determined the number of eggs remaining after breakfast. Now, I need to calculate how much Janet earns daily at the farmers' market with the remaining eggs.\nAction 2: Calculate[\n```python\neggs_remaining = 13\nprice_per_egg = 2\nearnings_per_day = eggs_remaining * price_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining = 13\nprice_per_egg = 2\nearnings_per_day = eggs_remaining * price_per_egg\n```\nExecution Status: Done\nOutput: earnings_per_day = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_remaining = eggs_laid_per_day - eggs_consumed\n```\n]",
        "I need to calculate how much money Janet makes from selling the remaining eggs at the farmers' market daily.\nAction 2: Calculate[\n```python\neggs_remaining = 13\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\n```\n]\nObservation 2: \n```python\neggs_remaining = 13\nprice_per_egg = 2\nmoney_made_daily = eggs_remaining * price_per_egg\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Calculate[\n```python\neggs_remaining = 13  # from previous calculation\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\n]\nObservation 2: \n```python\neggs_remaining = 13\negg_price = 2\nmoney_per_day = eggs_remaining * egg_price\nanswer = money_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day by selling the remaining duck eggs at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        "Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = ReflexionReActMathStrategy(llm=llm, testing=True)
    out = strategy.generate_react(
        question=question,
        key=key,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        additional_keys={},
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
    assert thought_metrics == PromptInfo(
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


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    assert strategy.halting_condition(3, "correct_answer", "incorrect_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    assert strategy.halting_condition(3, "correct_answer", "correct_answer") == False

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=3)
    assert strategy.halting_condition(4, "correct_answer", "incorrect_answer") == True


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActMathStrategy reflect_condition."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    llm = MockLLM("gpt-3.5-turbo", responses=["1"])
    strategy = ReflexionReActMathStrategy(llm=llm)
    out = strategy.reflect_condition(
        finished=False,
        answer="answer = 5",
        scratchpad="",
        idx=1,
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
