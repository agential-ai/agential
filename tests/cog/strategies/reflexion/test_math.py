"""Unit tests for Reflexion Math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.strategies.reflexion.math import (
    ReflexionCoTMathStrategy,
    ReflexionReActMathStrategy,
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
    parse_math_action_cot,
    parse_math_action_react
)
from agential.cog.prompts.agent.reflexion import (
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
)
from agential.cog.prompts.benchmark.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)


def test_parse_math_action_cot() -> None:
    """Tests parse_math_action_cot."""
    action = "Finish the calculation```python\nresult = 5 + 3\n```"
    action_type, query = parse_math_action_cot(action)
    assert action_type == "Finish"
    assert query == "result = 5 + 3"

    action = "complete the task```python\nanswer = 10 * 2\n```"
    action_type, query = parse_math_action_cot(action)
    assert action_type == ""
    assert query == ""


def test_parse_math_action_react() -> None:
    """Tests parse_math_action_react."""
    action = "Calculate the sum```python\nsum = 4 + 6\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == "Calculate"
    assert query == "sum = 4 + 6"

    action = "Finish the operation```python\nresult = 7 - 2\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == "Finish"
    assert query == "result = 7 - 2"

    action = "complete the task```python\noutput = 10 / 2\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == ""
    assert query == ""


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTMathStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_out = "Let's calculate the total number of eggs she sells after breakfast and baking muffins. Then, we can find out how much she makes daily at the farmers' market."
    gt_scratchpad = ""
    responses = [
        "Let's calculate the total number of eggs she sells after breakfast and baking muffins. Then, we can find out how much she makes daily at the farmers' market.\nAction: Finish[\n```python\neggs_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\ntotal_eggs_sold = eggs_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ndaily_income = total_eggs_sold * price_per_egg\nanswer = daily_income\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTMathStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    responses = [
        'Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day\n```\n]'
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTMathStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert query == 'eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day'
    assert strategy._finished == False
    assert strategy._answer == ""
    assert (
        strategy._scratchpad
        == '\nAction: Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day\n```\n]'
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTMathStrategy generate_observation."""
    # Case 1: action_type is "Finish" and answer is correct.
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish", query="correct_answer", key="correct_answer"
    )
    assert is_correct == True
    assert obs == "Answer is CORRECT"
    assert "Observation: Answer is CORRECT" in strategy._scratchpad

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTMathStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish", query="incorrect_answer", key="correct_answer"
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in strategy._scratchpad

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTMathStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Calculate", query="some_query", key="correct_answer"
    )
    assert is_correct == False
    assert obs == "Invalid action type, please try again."
    assert "Observation: Invalid action type, please try again." in strategy._scratchpad


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTMathStrategy create_output_dict."""
    strategy = ReflexionCoTMathStrategy(llm=FakeListChatModel(responses=[]))

    # Setting a dummy answer for testing.
    strategy._answer = "correct_answer"

    # Test case 1: Correct answer.
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        obs="Observation: Answer is CORRECT",
        is_correct=True,
        reflections=[],
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "observation": "Observation: Answer is CORRECT",
        "answer": "correct_answer",
        "is_correct": True,
        "reflections": [],
    }
    assert output == expected_output

    # Test case 2: Incorrect answer.
    strategy._answer = "incorrect_answer"
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        obs="Observation: Answer is INCORRECT",
        is_correct=False,
        reflections=[],
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "observation": "Observation: Answer is INCORRECT",
        "answer": "incorrect_answer",
        "is_correct": False,
        "reflections": [],
    }
    assert output == expected_output


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTMathStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm, max_trials=3)

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == True

    strategy._answer = "correct_answer"
    assert strategy.halting_condition(2, "correct_answer") == True

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(2, "correct_answer") == False


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTMathStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm, max_trials=3)

    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 1: Reset everything.
    strategy.reset()
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""

    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 2: Reset only scratchpad.
    strategy.reset(only_scratchpad=True)
    assert strategy._scratchpad == ""
    assert strategy._finished == True
    assert strategy._answer == "Some answer"


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTMathStrategy reflect."""


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTMathStrategy reflect_condition."""


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTMathStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionCoTSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionCoTTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionCoTGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionCoTSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionCoTTabMWPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActQAStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActGSM8KStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActQAStrategy generate."""


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActQAStrategy generate_action."""


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActQAStrategy generate_observation."""


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy create_output_dict."""


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy react_create_output_dict."""


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy halting_condition."""


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy react_halting_condition."""


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActQAStrategy reset."""


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActQAStrategy reflect."""


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActQAStrategy reflect_condition."""


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActQAStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReflexionReActGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionReActSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionReActTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionReActGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionReActSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionReActTabMWPStrategy)
