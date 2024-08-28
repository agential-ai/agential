"""Unit tests for Self-Refine math strategies."""

from agential.cog.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT
from agential.cog.self_refine.output import SelfRefineOutput, SelfRefineStepOutput
from agential.cog.self_refine.prompts import (
    GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
    GSM8K_REFINE_FEWSHOT_EXAMPLES,
    SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
    SELF_REFINE_INSTRUCTION_GSM8K,
    SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
)
from agential.cog.self_refine.strategies.math import (
    SelfRefineGSM8KStrategy,
    SelfRefineMathStrategy,
    SelfRefineSVAMPStrategy,
    SelfRefineTabMWPStrategy,
)
from agential.llm.llm import MockLLM, Response


def test_init() -> None:
    """Test SelfRefineMathStrategy initialization."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineMathStrategy(llm=llm, patience=3)
    assert strategy.llm == llm
    assert strategy.patience == 3
    assert strategy.testing == False
    assert strategy._prev_answer == ""
    assert strategy.patience_counter == 0


def test_generate() -> None:
    """Test SelfRefineMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    responses = [
        "eggs_per_day = 16\neggs_breakfast = 3\neggs_to_bake = 4933828\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\ntotal_eggs_sold = eggs_remain - eggs_to_bake\n\nmoney_made = eggs_sold_price * total_eggs_sold\nanswer = money_made",
        "The issue with the code is that the calculation for `total_eggs_sold` is incorrect. Instead of subtracting `eggs_to_bake` from `eggs_remain`, it should actually be subtracted from `eggs_per_day` to determine the number of eggs available for sale. Additionally, the variable `eggs_to_bake` is not necessary for the calculation of the money made at the farmers' market. \n\nHere is the corrected code:\n\n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\ntotal_eggs_sold = eggs_remain - eggs_to_bake\n\nmoney_made = eggs_sold_price * total_eggs_sold\nanswer = money_made\n``` \n\nAfter making these changes, the code should correctly calculate the amount of money Janet makes at the farmers' market every day.",
        "```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\nmoney_made = eggs_sold_price * eggs_remain\nanswer = money_made\n```",
        "There are no apparent issues with the code provided for this question. It correctly calculates the number of eggs Janet has to sell every day and then calculates the amount of money she makes by selling those eggs at the farmers' market.",
        "The code provided is correct and does not have any problems.",
        "There is no problem with the provided code.",
        "The provided code is correct and does not have any issues.",
    ]

    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineMathStrategy(llm=llm, patience=3)

    out = strategy.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt=SELF_REFINE_INSTRUCTION_GSM8K,
        critique_examples=GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        critique_prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        refine_examples=GSM8K_REFINE_FEWSHOT_EXAMPLES,
        refine_prompt=SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        additional_keys={},
        critique_additional_keys={},
        refine_additional_keys={},
        max_interactions=3,
        reset=True,
    )

    assert out == SelfRefineOutput(
        additional_info=[
            SelfRefineStepOutput(
                answer="eggs_per_day = 16\neggs_breakfast = 3\neggs_to_bake = 4933828\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\ntotal_eggs_sold = eggs_remain - eggs_to_bake\n\nmoney_made = eggs_sold_price * total_eggs_sold\nanswer = money_made",
                critique="The issue with the code is that the calculation for `total_eggs_sold` is incorrect. Instead of subtracting `eggs_to_bake` from `eggs_remain`, it should actually be subtracted from `eggs_per_day` to determine the number of eggs available for sale. Additionally, the variable `eggs_to_bake` is not necessary for the calculation of the money made at the farmers' market. \n\nHere is the corrected code:\n\n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\ntotal_eggs_sold = eggs_remain - eggs_to_bake\n\nmoney_made = eggs_sold_price * total_eggs_sold\nanswer = money_made\n``` \n\nAfter making these changes, the code should correctly calculate the amount of money Janet makes at the farmers' market every day.",
                answer_response=Response(
                    input_text="",
                    output_text="eggs_per_day = 16\neggs_breakfast = 3\neggs_to_bake = 4933828\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\ntotal_eggs_sold = eggs_remain - eggs_to_bake\n\nmoney_made = eggs_sold_price * total_eggs_sold\nanswer = money_made",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="The issue with the code is that the calculation for `total_eggs_sold` is incorrect. Instead of subtracting `eggs_to_bake` from `eggs_remain`, it should actually be subtracted from `eggs_per_day` to determine the number of eggs available for sale. Additionally, the variable `eggs_to_bake` is not necessary for the calculation of the money made at the farmers' market. \n\nHere is the corrected code:\n\n```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\ntotal_eggs_sold = eggs_remain - eggs_to_bake\n\nmoney_made = eggs_sold_price * total_eggs_sold\nanswer = money_made\n``` \n\nAfter making these changes, the code should correctly calculate the amount of money Janet makes at the farmers' market every day.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            SelfRefineStepOutput(
                answer="eggs_per_day = 16\neggs_breakfast = 3\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\nmoney_made = eggs_sold_price * eggs_remain\nanswer = money_made",
                critique="There are no apparent issues with the code provided for this question. It correctly calculates the number of eggs Janet has to sell every day and then calculates the amount of money she makes by selling those eggs at the farmers' market.",
                answer_response=Response(
                    input_text="",
                    output_text="```python\neggs_per_day = 16\neggs_breakfast = 3\neggs_sold_price = 2\n\neggs_remain = eggs_per_day - eggs_breakfast\nmoney_made = eggs_sold_price * eggs_remain\nanswer = money_made\n```",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="There are no apparent issues with the code provided for this question. It correctly calculates the number of eggs Janet has to sell every day and then calculates the amount of money she makes by selling those eggs at the farmers' market.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
            SelfRefineStepOutput(
                answer="The code provided is correct and does not have any problems.",
                critique="There is no problem with the provided code.",
                answer_response=Response(
                    input_text="",
                    output_text="The code provided is correct and does not have any problems.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
                critique_response=Response(
                    input_text="",
                    output_text="There is no problem with the provided code.",
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    prompt_cost=1.5e-05,
                    completion_cost=3.9999999999999996e-05,
                    total_cost=5.4999999999999995e-05,
                    prompt_time=0.5,
                ),
            ),
        ]
    )


def test_generate_answer() -> None:
    """Tests SelfRefineMathStrategy generate_answer."""
    llm = MockLLM("gpt-3.5-turbo", responses=["```python\nresult = 42\n```"])
    strategy = SelfRefineMathStrategy(llm=llm)
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"

    answer, out = strategy.generate_answer(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt=SELF_REFINE_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert answer == "result = 42"
    assert out == Response(
        input_text="",
        output_text="```python\nresult = 42\n```",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_generate_critique() -> None:
    """Tests SelfRefineMathStrategy generate_critique."""
    gt_critique = "The error in the code is that the result is hardcoded as 42 without actually calculating the total number of bolts needed for the robe. The code should calculate the total number of bolts required based on the information given in the question. Let's correct this:\n\n```python\nblue_bolts = 2\nwhite_bolts = blue_bolts / 2\ntotal_bolts = blue_bolts + white_bolts\nresult = total_bolts\n``` \n\nThis code snippet will correctly calculate the total number of bolts needed for the robe."
    responses = [
        "The error in the code is that the result is hardcoded as 42 without actually calculating the total number of bolts needed for the robe. The code should calculate the total number of bolts required based on the information given in the question. Let's correct this:\n\n```python\nblue_bolts = 2\nwhite_bolts = blue_bolts / 2\ntotal_bolts = blue_bolts + white_bolts\nresult = total_bolts\n``` \n\nThis code snippet will correctly calculate the total number of bolts needed for the robe."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineMathStrategy(llm=llm)
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    answer = "result = 42"

    critique, finished, out = strategy.generate_critique(
        question=question,
        examples=GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert critique == gt_critique
    assert strategy._prev_answer == answer
    assert strategy.patience_counter == 0
    assert finished == False
    assert out == Response(
        input_text="",
        output_text="The error in the code is that the result is hardcoded as 42 without actually calculating the total number of bolts needed for the robe. The code should calculate the total number of bolts required based on the information given in the question. Let's correct this:\n\n```python\nblue_bolts = 2\nwhite_bolts = blue_bolts / 2\ntotal_bolts = blue_bolts + white_bolts\nresult = total_bolts\n``` \n\nThis code snippet will correctly calculate the total number of bolts needed for the robe.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )

    # Test early stopping.
    gt_critique = "The error in the code is that the result is hardcoded as 42 without actually calculating the total number of bolts needed for the robe. The code should calculate the total number of bolts required based on the information given in the question. Let's correct this:\n\n```python\nblue_bolts = 2\nwhite_bolts = blue_bolts / 2\ntotal_bolts = blue_bolts + white_bolts\nresult = total_bolts\n``` \n\nThis code snippet will correctly calculate the total number of bolts needed for the robe."
    answer1 = "result = 42"
    responses = [
        "The error in the code is that the result is hardcoded as 42 without actually calculating the total number of bolts needed for the robe. The code should calculate the total number of bolts required based on the information given in the question. Let's correct this:\n\n```python\nblue_bolts = 2\nwhite_bolts = blue_bolts / 2\ntotal_bolts = blue_bolts + white_bolts\nresult = total_bolts\n``` \n\nThis code snippet will correctly calculate the total number of bolts needed for the robe."
    ]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineMathStrategy(llm=llm, patience=1)
    strategy._prev_answer = "result = 42"
    critique, finished, out = strategy.generate_critique(
        question=question,
        examples=GSM8K_CRITIQUE_FEWSHOT_EXAMPLES,
        answer=answer1,
        prompt=SELF_REFINE_CRITIQUE_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert critique == gt_critique
    assert strategy.patience_counter == 1
    assert strategy._prev_answer == "result = 42"
    assert finished
    assert out == Response(
        input_text="",
        output_text="The error in the code is that the result is hardcoded as 42 without actually calculating the total number of bolts needed for the robe. The code should calculate the total number of bolts required based on the information given in the question. Let's correct this:\n\n```python\nblue_bolts = 2\nwhite_bolts = blue_bolts / 2\ntotal_bolts = blue_bolts + white_bolts\nresult = total_bolts\n``` \n\nThis code snippet will correctly calculate the total number of bolts needed for the robe.",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_update_answer_based_on_critique() -> None:
    """Tests SelfRefineMathStrategy update_answer_based_on_critique."""
    responses = ["```python\nresult = 43\n```"]
    llm = MockLLM("gpt-3.5-turbo", responses=responses)
    strategy = SelfRefineMathStrategy(llm=llm)
    question = "Sample question"
    answer = "result = 42"
    critique = "Critique: Your solution is incorrect."

    new_answer, out = strategy.update_answer_based_on_critique(
        question=question,
        examples=GSM8K_REFINE_FEWSHOT_EXAMPLES,
        answer=answer,
        critique=critique,
        prompt=SELF_REFINE_REFINE_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert new_answer == "result = 43"
    assert out == Response(
        input_text="",
        output_text="```python\nresult = 43\n```",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost=1.5e-05,
        completion_cost=3.9999999999999996e-05,
        total_cost=5.4999999999999995e-05,
        prompt_time=0.5,
    )


def test_halting_condition() -> None:
    """Tests SelfRefineMathStrategy halting_condition."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineMathStrategy(llm=llm, patience=2)

    # Initially, halting condition should be False.
    assert strategy.halting_condition(False) is False

    # Simulate the halting condition being met.
    assert strategy.halting_condition(True)


def test_reset() -> None:
    """Tests SelfRefineMathStrategy reset."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    strategy = SelfRefineMathStrategy(llm=llm, patience=2)

    strategy._prev_answer = "result = 42"
    strategy.patience_counter = 1
    strategy.reset()
    assert strategy._prev_answer == ""
    assert strategy.patience_counter == 0


def test_instantiate_strategies() -> None:
    """Test instantiate all Math strategies."""
    llm = MockLLM("gpt-3.5-turbo", responses=[])
    assert isinstance(SelfRefineGSM8KStrategy(llm=llm), SelfRefineGSM8KStrategy)
    assert isinstance(SelfRefineSVAMPStrategy(llm=llm), SelfRefineSVAMPStrategy)
    assert isinstance(SelfRefineTabMWPStrategy(llm=llm), SelfRefineTabMWPStrategy)
