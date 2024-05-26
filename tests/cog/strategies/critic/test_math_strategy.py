"""Unit tests for CRITIC math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel

from agential.cog.strategies.critic.math_strategy import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
    CriticMathStrategy,
)
from agential.cog.prompts.critic import (
    GSM8K_FEWSHOT_EXAMPLES_POT,
    CRITIC_POT_INSTRUCTION_GSM8K,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
)


def test_generate() -> None:
    """Tests CriticMathStrategy generate."""
    llm = FakeListChatModel(responses=["Generated answer\n```python\n42\n```"])
    strategy = CriticMathStrategy(llm=llm)
    question = "What is 6 multiplied by 7?"

    result = strategy.generate(
        question, 
        examples=GSM8K_FEWSHOT_EXAMPLES_POT, 
        prompt=CRITIC_POT_INSTRUCTION_GSM8K, 
        additional_keys={}
    )

    assert result == "42"


def test_generate_critique() -> None:
    """Tests CriticMathStrategy generate_critique."""
    gt_result = 'The answer provided (40) is incorrect. The correct answer to the question "What is 6 multiplied by 7?" is 42, not 40. \n\n'
    responses = [
        'The answer provided (40) is incorrect. The correct answer to the question "What is 6 multiplied by 7?" is 42, not 40. \n\nHere\'s the corrected code:\n```python\nresult = 6 * 7\nanswer = result\n```'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticMathStrategy(llm=llm)

    question = "What is 6 multiplied by 7?"
    answer = "40"

    result, external_tool_info = strategy.generate_critique(
        idx=0,
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        additional_keys={},
        use_tool=False,
        max_interactions=5,
    )

    assert result == gt_result
    assert external_tool_info == {}

    # Test with tool.
    gt_result = "1. The revenue from selling eggs should be a positive number, -9867630 < 0, which is not reasonable.\n\n2. Let's check the code:\n\n- `total_eggs = 16` - This defines the total number of eggs laid by Janet's ducks per day.\n- `eaten_eggs = 3` - This represents the number of eggs Janet eats for breakfast.\n- `baked_eggs = 4933828` - This represents the number of eggs Janet uses to bake muffins for her friends daily.\n- `sold_eggs = total_eggs - eaten_eggs - baked_eggs` - This calculates the number of eggs Janet has left to sell at the farmers' market.\n- `dollars_per_egg = 2` - This represents the selling price of each fresh duck egg.\n- `answer = sold_eggs * dollars_per_egg` - This calculates the total revenue from selling eggs at the farmers' market.\n\nThe issue with the code is that the calculation for `sold_eggs` is incorrect. Janet should only sell the eggs that are left after she eats some for breakfast and uses some for baking. \n\n"
    responses = [
        "1. The revenue from selling eggs should be a positive number, -9867630 < 0, which is not reasonable.\n\n2. Let's check the code:\n\n- `total_eggs = 16` - This defines the total number of eggs laid by Janet's ducks per day.\n- `eaten_eggs = 3` - This represents the number of eggs Janet eats for breakfast.\n- `baked_eggs = 4933828` - This represents the number of eggs Janet uses to bake muffins for her friends daily.\n- `sold_eggs = total_eggs - eaten_eggs - baked_eggs` - This calculates the number of eggs Janet has left to sell at the farmers' market.\n- `dollars_per_egg = 2` - This represents the selling price of each fresh duck egg.\n- `answer = sold_eggs * dollars_per_egg` - This calculates the total revenue from selling eggs at the farmers' market.\n\nThe issue with the code is that the calculation for `sold_eggs` is incorrect. Janet should only sell the eggs that are left after she eats some for breakfast and uses some for baking. \n\nHere's a corrected solution:\n\n```python\ntotal_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\n\n# Calculate the number of eggs left to sell\nremaining_eggs = total_eggs - eaten_eggs - baked_eggs\n\ndollars_per_egg = 2\n\n# Calculate the revenue from selling eggs\ntotal_revenue = remaining_eggs * dollars_per_egg\n\nanswer = total_revenue\n```"
    ]
    llm = FakeListChatModel(
        responses=responses
    )
    strategy = CriticMathStrategy(llm=llm)
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    answer = 'total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg'

    result, external_tool_info = strategy.generate_critique(
        idx=0,
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
        additional_keys={},
        use_tool=True,
        max_interactions=5,
    )

    assert result == gt_result
    assert external_tool_info["execution_status"] == "Done"
    assert external_tool_info["code_answer"] == -9867630

    # Test increment patience counter and early stopping.
    gt_result = "The output of -9867630 is incorrect because the amount of money Janet makes should be a positive number, as she is selling eggs at the farmers' market. \n\nLet's analyze the code:\n- `total_eggs = 16`: This represents the total number of eggs the ducks lay per day, which is correct.\n- `eaten_eggs = 3`: This represents the number of eggs Janet eats for breakfast, which is also correct.\n- `baked_eggs = 4933828`: This number seems unusually high for baking muffins daily. It might be a mistake in the code.\n- `sold_eggs = total_eggs - eaten_eggs - baked_eggs`: This calculates the number of eggs sold, but the calculation may be incorrect due to the high number of baked eggs.\n- `dollars_per_egg = 2`: This represents the selling price per fresh duck egg, which is correct.\n- `answer = sold_eggs * dollars_per_egg`: This calculates the total amount of money made from selling eggs, but the previous calculations might be incorrect.\n\nTo correct the code and ensure an accurate calculation of the money Janet makes every day at the farmers' market, we need to revisit the logic of how many eggs she bakes for muffins and how many she sells. \n\n"
    gt_answer_history = [{'answer': 'total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg', 'external_tool_info': {'execution_status': 'Done', 'code_answer': -9867630}}]
    responses = [
        "The output of -9867630 is incorrect because the amount of money Janet makes should be a positive number, as she is selling eggs at the farmers' market. \n\nLet's analyze the code:\n- `total_eggs = 16`: This represents the total number of eggs the ducks lay per day, which is correct.\n- `eaten_eggs = 3`: This represents the number of eggs Janet eats for breakfast, which is also correct.\n- `baked_eggs = 4933828`: This number seems unusually high for baking muffins daily. It might be a mistake in the code.\n- `sold_eggs = total_eggs - eaten_eggs - baked_eggs`: This calculates the number of eggs sold, but the calculation may be incorrect due to the high number of baked eggs.\n- `dollars_per_egg = 2`: This represents the selling price per fresh duck egg, which is correct.\n- `answer = sold_eggs * dollars_per_egg`: This calculates the total amount of money made from selling eggs, but the previous calculations might be incorrect.\n\nTo correct the code and ensure an accurate calculation of the money Janet makes every day at the farmers' market, we need to revisit the logic of how many eggs she bakes for muffins and how many she sells. \n\nHere's a revised solution:\n```python\ntotal_eggs = 16\neaten_eggs = 3\nbaked_eggs = 6  # Let's assume Janet bakes 6 eggs for muffins daily\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\ndaily_income = sold_eggs * dollars_per_egg\n\nanswer = daily_income\n``` \n\nBy adjusting the number of eggs baked for muffins to a more reasonable amount (e.g., 6 eggs), the calculation should yield a positive value for the daily income Janet makes at the farmers' market."    ]
    llm = FakeListChatModel(
        responses=responses
    )
    strategy = CriticMathStrategy(llm=llm)
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    answer = 'total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg'

    strategy._prev_code_answer = -9867630
    strategy.patience_counter = 1
    result, external_tool_info = strategy.generate_critique(
        idx=1,
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC,
        answer=answer,
        critique="",
        prompt=CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
        additional_keys={},
        use_tool=True,
        max_interactions=5,
    )

    assert result == gt_result
    assert external_tool_info["execution_status"] == "Done"
    assert external_tool_info["code_answer"] == -9867630
    assert strategy._answer_history == gt_answer_history
    assert strategy._prev_code_answer == -9867630
    assert strategy.patience_counter == 2
    assert strategy._halt is True



def test_create_output_dict() -> None:
    """Tests CriticMathStrategy create_output_dict."""
    llm = FakeListChatModel(responses=[])
    strategy = CriticMathStrategy(llm=llm)

    answer = -9867630
    critique = "The answer is correct."
    external_tool_info = {"execution_status": "Done", "code_answer": -9867630}

    result = strategy.create_output_dict(answer, critique, external_tool_info)

    assert result["code"] == answer
    assert result["critique"] == critique
    assert result["execution_status"] == external_tool_info['execution_status']
    assert result["code_answer"] ==external_tool_info['code_answer']


def test_update_answer_based_on_critique() -> None:
    """Tests CriticMathStrategy update_answer_based_on_critique."""

    # Test without tool.
    gt_new_answer = 'total_eggs_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = total_eggs_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\n\ntotal_earnings_per_day = eggs_sold * price_per_egg\nanswer = total_earnings_per_day'
    responses = [
        'total_eggs_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = total_eggs_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\n\ntotal_earnings_per_day = eggs_sold * price_per_egg\nanswer = total_earnings_per_day'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = CriticMathStrategy(llm=llm)

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    answer = 'total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg'
    critique = "The code correctly calculates the number of eggs sold at the farmers' market daily and then multiplies that by the price per egg to determine the total earnings. The answer given by the code would be the correct amount of money Janet makes every day at the farmers' market. \n\nTherefore, there doesn't seem to be any problem with the above code."

    new_answer = strategy.update_answer_based_on_critique(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        answer=answer,
        critique=critique,
        prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        additional_keys={},
        external_tool_info={},
    )

    assert new_answer == gt_new_answer

    # Test with tool.
    gt_new_answer = 'total_eggs_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = total_eggs_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\n\ntotal_earnings_per_day = eggs_sold * price_per_egg\nanswer = total_earnings_per_day'
    answer = 'total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg'
    critique = "The problem with the above code is that the calculation for `baked_eggs` seems incorrect. It is stated that Janet bakes muffins for her friends every day with 4933828 eggs, which seems like an excessive amount. This is likely causing the negative result in the final calculation.\n\nTo fix this issue and provide a more reasonable calculation, we need to adjust the amount of eggs used for baking muffins to a more realistic number. Let's assume she bakes 12 muffins each day, which would require 12 eggs. \n\n"
    external_tool_info = {'execution_status': 'Done', 'code_answer': -9867630}

    new_answer = strategy.update_answer_based_on_critique(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        answer=answer,
        critique=critique,
        prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        additional_keys={},
        external_tool_info=external_tool_info,
    )

    assert new_answer == gt_new_answer


def test_halting_condition() -> None:
    """Tests CriticMathStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = CriticMathStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition("") is False

    # Simulate the halting condition being met.
    strategy._halt = True
    assert strategy.halting_condition("") is True


def test_reset() -> None:
    """Tests CriticMathStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = CriticMathStrategy(llm=llm, patience=2)

    # Simulate some state
    strategy._answer_history = [{"answer": "some_answer", "external_tool_info": {}}]
    strategy._prev_code_answer = "42"
    strategy.patience_counter = 1
    strategy._halt = True

    # Reset the strategy
    strategy.reset()

    # Assert that all states are reset
    assert strategy._answer_history == []
    assert strategy._prev_code_answer is None
    assert strategy.patience_counter == 0
    assert strategy._halt is False
    

def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""