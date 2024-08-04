"""Unit tests for LATS Math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.docstore.wikipedia import Wikipedia

from agential.cog.fewshots.gsm8k import GSM8K_FEWSHOT_EXAMPLES_REACT
from agential.cog.lats.node import Node
from agential.cog.lats.prompts import (
    GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
    LATS_INSTRUCTION_GSM8K,
    LATS_REFLECT_INSTRUCTION_GSM8K,
    GSM8K_FEWSHOT_EXAMPLES_LATS_VALUE,
    LATS_VALUE_INSTRUCTION_GSM8K,
)
from agential.cog.lats.strategies.math import (
    LATSMathStrategy,
    LATSSVAMPStrategy,
    LATSTabMWPStrategy,
    LATSMathStrategy,
    parse_math_action,
    parse_math_value,
    get_node_trajectory_math
)
from agential.cog.react.output import ReActOutput
from agential.utils.docstore import DocstoreExplorer


def test_get_node_trajectory_math() -> None:
    """Tests the get_node_trajectory_math() function."""
    root = Node(
        state=ReActOutput(
            **{
                "thought": "Root thought",
                "action_type": "",
                "query": "",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        )
    )
    child1 = Node(
        state=ReActOutput(
            **{
                "thought": "Child1 thought",
                "action_type": "Lookup",
                "query": "topic",
                "observation": "",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        parent=root,
    )
    child2 = Node(
        state=ReActOutput(
            **{
                "thought": "Child2 thought",
                "action_type": "Finish",
                "query": "answer",
                "observation": "Answer correct",
                "answer": "",
                "external_tool_info": {},
            }
        ),
        parent=child1,
    )

    expected_trajectory = '\nThought 1: Child1 thought\nAction 1: Lookup[\n```python\ntopic\n```\n]\nThought 2: Child2 thought\nAction 2: Finish[\n```python\nanswer\n```\n]\nObservation 2: Answer correct'
    assert get_node_trajectory_math(child2) == expected_trajectory

    # Test root node.
    root = Node()
    assert get_node_trajectory_math(root) == ""


def test_parse_math_action():
    """Test the parse_math_action function."""
    test_cases = [
        {
            "input": "Calculate[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Calculate", "def add(a, b): return a + b"),
        },
        {
            "input": "FINISH[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Finish", "assert add(2, 3) == 5"),
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
            "input": "Calculate[```python\n \n```]",
            "expected": ("Calculate", ""),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
    ]

    for case in test_cases:
        result = parse_math_action(case["input"])
        assert result == case["expected"], f"Failed for input: {case['input']}"


def test_parse_math_value():
    """Test the parse_math_value function."""
    # Test valid value strings.
    valid_input = (
        "Some text. Explanation: This is the explanation. Correctness score: 5"
    )
    assert parse_math_value(valid_input) == ("This is the explanation.", 5)

    # Test invalid value strings.
    assert parse_math_value("No explanation or score") == ("Explanation not found", 0)
    assert parse_math_value("Explanation: Only explanation") == (
        "Explanation not found",
        0,
    )
    assert parse_math_value("Correctness score: 5") == ("Explanation not found", 0)

    # Test edge cases.
    assert parse_math_value("Explanation: Empty. Correctness score: 0") == ("Empty.", 0)
    assert parse_math_value(
        "Explanation: Multi-line\nexplanation. Correctness score: 10"
    ) == ("Multi-line\nexplanation.", 10)

    # Test with unexpected format.
    assert parse_math_value("Explanation: Tricky: score. Correctness score: 7") == (
        "Tricky: score.",
        7,
    )


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=[])
    docstore = DocstoreExplorer(Wikipedia())
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


def test_initialize() -> None:
    """Test the initialize method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm)

    node = strategy.initialize()

    assert strategy.root == node
    assert strategy.root is not None
    assert isinstance(strategy.root, Node)
    assert strategy.root.state.thought == ""
    assert strategy.root.state.action_type == ""
    assert strategy.root.state.query == ""
    assert strategy.root.state.observation == ""
    assert strategy.root.state.external_tool_info == {}



def test_generate_thought() -> None:
    """Test the generate_thought method."""
    llm = FakeListChatModel(
        responses=[
            "I should search for information about the topic."
        ]
    )
    strategy = LATSMathStrategy(llm=llm)

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    trajectory = "Previous thought"
    reflections = "Reflection 1\nReflection 2"
    depth = 1
    prompt = "Generate a thought"
    additional_keys = {"key": "value"}

    updated_trajectory, thought = strategy.generate_thought(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )

    assert thought == "I should search for information about the topic."
    assert (
        updated_trajectory
        == "Previous thought\nThought 2: I should search for information about the topic."
    )


def test_generate_action() -> None:
    """Test the generate_action method."""
    llm = FakeListChatModel(responses=["Calculate[```python\nresult = 2 + 2\n```]"])
    strategy = LATSMathStrategy(llm=llm)

    question = "What is 2 + 2?"
    examples = "Example 1\nExample 2"
    trajectory = "Thought 1: I need to calculate 2 + 2."
    reflections = "Reflection 1\nReflection 2"
    depth = 0
    prompt = "Generate an action"
    additional_keys = {"key": "value"}

    trajectory, action_type, query = strategy.generate_action(
        question, examples, trajectory, reflections, depth, prompt, additional_keys
    )
    
    assert trajectory == "Thought 1: I need to calculate 2 + 2.\nAction 1: Calculate[\n```python\nresult = 2 + 2\n```\n]"
    assert action_type == "Calculate"
    assert query == "result = 2 + 2"



def test_generate_observation() -> None:
    """Test the generate_observation method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm)

    key = "4"
    trajectory = "Previous trajectory"

    # Test Finish action.
    finish_result = strategy.generate_observation(key, "Finish", "4", trajectory, 1)
    assert finish_result == ('Previous trajectory\nObservation 2: Answer is INCORRECT', 0, 'Answer is INCORRECT', True, {'execution_status': 'Done', 'code_answer': None})

    # Test Calculate action.
    calculate_result = strategy.generate_observation(
        key, "Calculate", "result = 2 + 2", trajectory, 2
    )
    assert calculate_result == ('Previous trajectory\nObservation 3: \n```python\nresult = 2 + 2\n```\nExecution Status: Done\nOutput: answer = None', 0, '\n```python\nresult = 2 + 2\n```\nExecution Status: Done\nOutput: answer = None', False, {'execution_status': 'Done', 'code_answer': None})

    # Test invalid action.
    invalid_result = strategy.generate_observation(
        key, "Invalid", "query", trajectory, 3
    )
    assert invalid_result == ('Previous trajectory\nObservation 4: Invalid Action. Valid Actions are Calculate[code] and Finish[answer].', 0, 'Invalid Action. Valid Actions are Calculate[code] and Finish[answer].', False, {'execution_status': '', 'code_answer': ''})


def test_generate() -> None:
    """Test the generate method."""
    gt_states = [
        ReActOutput(thought="I need to calculate how much money Janet makes daily at the farmers' market.", action_type='Calculate', query='eggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day', observation='\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': -9867630}),
        ReActOutput(thought="I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.", action_type='Calculate', query='eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg', observation='\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\nExecution Status: Done\nOutput: answer = -9867630', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': -9867630}),
        ReActOutput(thought="First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.", action_type='Calculate', query='eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins', observation='\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: answer = None', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': None}),
        ReActOutput(thought='First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.', action_type='Calculate', query='eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast', observation='\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': None}),
        ReActOutput(thought='First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.', action_type='Calculate', query='eggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins', observation='\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\nExecution Status: Done\nOutput: answer = None', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': None}),
    ]
    responses = [
        "I need to calculate how much money Janet makes daily at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 2: The calculation resulted in a negative value, which doesn't make sense for money made. I need to review the calculation.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 9867650\nThought 3: Janet makes $9867650 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 9867650\n```\n]\nObservation 3:\n```python\nanswer = 9867650\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\n]',
        "I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867649\nThought 2: The calculation resulted in a negative number, which doesn't make sense in this context. I need to review the subtraction steps.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\n]',
        "First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: The negative result indicates an error in the calculations. I need to correct the computation to determine the correct number of eggs left for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933807\nThought 3: Another error occurred in the calculation. I need to correct the computation to determine the correct number of eggs available for sale.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = -9867614\nThought 4: The negative result indicates that there was an error in the calculation. I need to correct the computation to determine the accurate daily profit from selling duck eggs at the farmers' market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = 9867614\nThought 5: Janet makes $9867614 every day at the farmers' market.\nAction 5: Finish[\n```python\nprofit_per_day = 9867614\n```\n]\nObservation 5: \n```python\nprofit_per_day = 9867614\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\n]',
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\nExecution Status: Done\nOutput: eggs_remaining_per_day = 13\nThought 2: Now, I need to find out how much Janet earns from selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]',
        "First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 2: The calculation is incorrect because Janet cannot have negative eggs to sell. I need to review the subtraction.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = 5\nThought 3: Janet sells 5 fresh duck eggs every day at the farmers' market.\nAction 3: Finish[\n```python\neggs_sold = 5\n```\n]\nObservation 3:\n```python\neggs_sold = 5\n```",
        'Calculate[\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\n]',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSMathStrategy(llm=llm)

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    root = strategy.initialize()

    children_nodes = strategy.generate(
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
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0

    # Test generate with reflections.
    # gt_states = [
    #     ReActOutput(
    #         thought="I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and crimes of violence outside the ring",
    #         action_type="Search",
    #         query="best kickboxer controversies violence",
    #         observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #         answer="",
    #         external_tool_info={
    #             "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #             "lookup_result": "",
    #         },
    #     ),
    #     ReActOutput(
    #         thought="I need to search for the best kick boxer in the world and then look into his controversies related to unsportsmanlike conduct and crimes of violence",
    #         action_type="Search",
    #         query="best kick boxer in the world",
    #         observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #         answer="",
    #         external_tool_info={
    #             "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #             "lookup_result": "",
    #         },
    #     ),
    #     ReActOutput(
    #         thought="I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring",
    #         action_type="Search",
    #         query="best kick boxer in the world controversies",
    #         observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #         answer="",
    #         external_tool_info={
    #             "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #             "lookup_result": "",
    #         },
    #     ),
    #     ReActOutput(
    #         thought="I need to search for the best kickboxer in the world who has been involved in controversies regarding unsportsmanlike conduct and crimes of violence outside the ring",
    #         action_type="Search",
    #         query="best kickboxer controversies",
    #         observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #         answer="",
    #         external_tool_info={
    #             "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #             "lookup_result": "",
    #         },
    #     ),
    #     ReActOutput(
    #         thought="I need to search for the best kick boxer in the world and his controversies regarding unsportsmanlike conducts and crimes of violence",
    #         action_type="Search",
    #         query="best kick boxer in the world controversies",
    #         observation="Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #         answer="",
    #         external_tool_info={
    #             "search_result": "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues.",
    #             "lookup_result": "",
    #         },
    #     ),
    # ]
    # responses = [
    #     "My reasoning for this question failed because I did not narrow down the search to focus on kick boxers and instead ended up with unrelated information",
    #     "My reasoning failed because I did not focus on gathering specific information related to the individual's kickboxing career and controversies, leading to an incorrect answer",
    #     "I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and crimes of violence outside the ring",
    #     "Search[best kickboxer controversies violence]\nObservation 1: Could not find [best kickboxer controversies violence]",
    #     "I need to search for the best kick boxer in the world and then look into his controversies related to unsportsmanlike conduct and crimes of violence",
    #     "Search[best kick boxer in the world]\nObservation 1: There have been several renowned kickboxers throughout history, such as Buakaw Banchamek, Ernesto Hoost, and Ramon Dekkers",
    #     "I need to search for the best kick boxer in the world who has been involved in controversies related to unsportsmanlike conduct and violence outside of the ring",
    #     "Search[best kick boxer in the world controversies]\nObservation 1: Could not find [best kick boxer in the world controversies]",
    #     "I need to search for the best kickboxer in the world who has been involved in controversies regarding unsportsmanlike conduct and crimes of violence outside the ring",
    #     "Search[best kickboxer controversies]\nObservation 1: Could not find [best kickboxer controversies]",
    #     "I need to search for the best kick boxer in the world and his controversies regarding unsportsmanlike conducts and crimes of violence",
    #     "Search[best kick boxer in the world controversies]\nObservation 1: Could not find [best kick boxer in the world controversies]",
    # ]
    # llm = FakeListChatModel(responses=responses)
    # strategy = LATSMathStrategy(llm=llm)
    # strategy.docstore.search = (
    #     lambda x: "Badr Hari, known as the 'Golden Boy', is a Dutch-Moroccan kickboxer who has been involved in several controversies and legal issues."
    # )
    # strategy.failed_trajectories = [
    #     {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
    #     {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
    #     {
    #         "trajectory": "Failed trajectory 1",
    #         "final_answer": "Incorrect answer 1",
    #     },  # Duplicate, should be ignored
    # ]

    # root = strategy.initialize()
    # children_nodes = strategy.generate(
    #     node=root,
    #     question=question,
    #     key=key,
    #     examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
    #     reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
    #     prompt=LATS_INSTRUCTION_GSM8K,
    #     reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
    #     additional_keys={},
    #     reflect_additional_keys={},
    # )
    # assert len(children_nodes) == 5
    # for gt_state, node in zip(gt_states, children_nodes):
    #     assert node.state == gt_state
    #     assert node.depth == 1
    #     assert node.reward == 0
    #     assert node.value == 0
    #     assert node.is_terminal is False
    #     assert node.visits == 0

    # Test case with a terminal child node (reward 0)
    # responses = [
    #     "I think the answer is Mike Tyson.",
    #     "Finish[Mike Tyson]",
    # ]
    # llm = FakeListChatModel(responses=responses)
    # strategy = LATSMathStrategy(llm=llm, n_samples=1)

    # root = strategy.initialize()
    # children_nodes = strategy.generate(
    #     node=root,
    #     question=question,
    #     key=key,
    #     examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
    #     reflect_examples=GSM8K_FEWSHOT_EXAMPLES_LATS_REFLECT,
    #     prompt=LATS_INSTRUCTION_GSM8K,
    #     reflect_prompt=LATS_REFLECT_INSTRUCTION_GSM8K,
    #     additional_keys={},
    #     reflect_additional_keys={},
    # )
    # assert len(children_nodes) == 1
    # assert children_nodes[0].state.thought == "I think the answer is Mike Tyson."
    # assert children_nodes[0].state.action_type == "Finish"
    # assert children_nodes[0].state.query == "Mike Tyson"
    # assert children_nodes[0].is_terminal
    # assert children_nodes[0].reward == 0


def test_select_node() -> None:
    """Test the select_node method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm)

    # Create a tree structure.
    root = Node(state={})
    child1 = Node(state={}, parent=root)
    child2 = Node(state={}, parent=root)
    grandchild1 = Node(state={}, parent=child1)
    grandchild2 = Node(state={}, parent=child1)

    root.children = [child1, child2]
    child1.children = [grandchild1, grandchild2]

    # Test selection of non-terminal node with highest UCT.
    child1.visits = 10
    child1.value = 0.6
    child2.visits = 5
    child2.value = 0.4
    selected_node = strategy.select_node(root)
    assert (
        selected_node == grandchild1
    )  # child2 should have higher UCT due to fewer visits

    # Test pruning of fully expanded terminal node.
    grandchild2.is_terminal = True
    grandchild2.reward = 0
    selected_node = strategy.select_node(root)
    assert selected_node == grandchild1

    # Test selection when all children are terminal.
    root = Node(state={})
    child1 = Node(state={}, parent=root)
    child2 = Node(state={}, parent=root)
    root.add_children([child1, child2])
    child1.is_terminal = True
    child2.is_terminal = True
    selected_node = strategy.select_node(root)
    assert selected_node == root


def test_expand_node() -> None:
    """Test the expand_node method."""
    gt_states = [
        ReActOutput(thought="I need to calculate how much money Janet makes daily at the farmers' market.", action_type='Calculate', query='eggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day', observation='\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\nExecution Status: Done\nOutput: answer = -9867630', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': -9867630}),
        ReActOutput(thought="I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.", action_type='Calculate', query='eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg', observation='\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\nExecution Status: Done\nOutput: answer = -9867630', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': -9867630}),
        ReActOutput(thought="First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.", action_type='Calculate', query='eggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins', observation='\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: answer = None', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': None}),
        ReActOutput(thought='First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.', action_type='Calculate', query='eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast', observation='\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\nExecution Status: Done\nOutput: answer = None', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': None}),
        ReActOutput(thought='First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.', action_type='Calculate', query='eggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins', observation='\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\nExecution Status: Done\nOutput: answer = None', answer='', external_tool_info={'execution_status': 'Done', 'code_answer': None}),
    ]

    responses = [
        "I need to calculate how much money Janet makes daily at the farmers' market.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867650\nThought 2: The calculation resulted in a negative value, which doesn't make sense for money made. I need to review the calculation.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nmoney_made_per_day = abs(money_made_per_day)  # Take the absolute value\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 9867650\nThought 3: Janet makes $9867650 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 9867650\n```\n]\nObservation 3:\n```python\nanswer = 9867650\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed = 3\neggs_used_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_consumed - eggs_used_muffins\nprice_per_egg = 2\nearnings_per_day = eggs_sold * price_per_egg\nanswer = earnings_per_day\n```\n]',
        "I need to calculate how much money Janet makes daily at the farmers' market by selling the remaining eggs after breakfast and baking muffins for her friends.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = -9867649\nThought 2: The calculation resulted in a negative number, which doesn't make sense in this context. I need to review the subtraction steps.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\nremaining_eggs = eggs_laid_per_day - eggs_for_breakfast\nprice_per_egg = 2\nmoney_made_per_day = remaining_eggs * price_per_egg\nanswer = money_made_per_day\n```\nExecution Status: Done\nOutput: answer = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3: \n```python\nanswer = 26\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nanswer = eggs_sold * price_per_egg\n```\n]',
        "First, I need to calculate the total number of eggs Janet has available to sell at the farmers' market after accounting for her breakfast consumption and muffin baking.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\n]\nObservation 1: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_remaining = -4933815\nThought 2: The negative result indicates an error in the calculations. I need to correct the computation to determine the correct number of eggs left for sale.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\n]\nObservation 2: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\n```\nExecution Status: Done\nOutput: eggs_for_sale = -4933807\nThought 3: Another error occurred in the calculation. I need to correct the computation to determine the correct number of eggs available for sale.\nAction 3: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\n]\nObservation 3: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = eggs_for_sale * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = -9867614\nThought 4: The negative result indicates that there was an error in the calculation. I need to correct the computation to determine the accurate daily profit from selling duck eggs at the farmers' market.\nAction 4: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\n]\nObservation 4: \n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\nmuffin_egg_usage = 4933828\neggs_remaining = eggs_laid_per_day - eggs_consumed_for_breakfast\neggs_for_sale = eggs_remaining - muffin_egg_usage\nprice_per_egg = 2\nprofit_per_day = abs(eggs_for_sale) * price_per_egg\n```\nExecution Status: Done\nOutput: profit_per_day = 9867614\nThought 5: Janet makes $9867614 every day at the farmers' market.\nAction 5: Finish[\n```python\nprofit_per_day = 9867614\n```\n]\nObservation 5: \n```python\nprofit_per_day = 9867614\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_available_to_sell = eggs_laid_per_day - eggs_consumed_for_breakfast - eggs_baked_into_muffins\n```\n]',
        "First, I need to calculate how many eggs Janet has left after eating three for breakfast every day.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_consumed_per_day = 3\neggs_remaining_per_day = eggs_laid_per_day - eggs_consumed_per_day\n```\nExecution Status: Done\nOutput: eggs_remaining_per_day = 13\nThought 2: Now, I need to find out how much Janet earns from selling the remaining eggs at the farmers' market.\nAction 2: Calculate[\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\n]\nObservation 2:\n```python\neggs_remaining_per_day = 13\nearnings_per_egg = 2\ndaily_earnings = eggs_remaining_per_day * earnings_per_egg\n```\nExecution Status: Done\nOutput: daily_earnings = 26\nThought 3: Janet makes $26 every day at the farmers' market.\nAction 3: Finish[\n```python\nanswer = 26\n```\n]\nObservation 3:\n```python\nanswer = 26\n```",
        'Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_remaining = eggs_laid_per_day - eggs_eaten_for_breakfast\n```\n]',
        "First, I need to calculate how many eggs Janet has left after eating three eggs for breakfast and baking muffins.\nAction 1: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 1:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = -4933815\nThought 2: The calculation is incorrect because Janet cannot have negative eggs to sell. I need to review the subtraction.\nAction 2: Calculate[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\n]\nObservation 2:\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_baked_into_muffins\n```\nExecution Status: Done\nOutput: eggs_sold = 5\nThought 3: Janet sells 5 fresh duck eggs every day at the farmers' market.\nAction 3: Finish[\n```python\neggs_sold = 5\n```\n]\nObservation 3:\n```python\neggs_sold = 5\n```",
        'Calculate[\n```python\neggs_per_day = 16\neggs_eaten_breakfast = 3\neggs_baked_in_muffins = 4933828\neggs_remaining = eggs_per_day - eggs_eaten_breakfast - eggs_baked_in_muffins\n```\n]',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = LATSMathStrategy(llm=llm)


    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    key = -9867630

    root = strategy.initialize()

    children_nodes = strategy.expand_node(
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
    for gt_state, node in zip(gt_states, children_nodes):
        assert node.state == gt_state
        assert node.depth == 1
        assert node.reward == 0
        assert node.value == 0
        assert node.is_terminal is False
        assert node.visits == 0
    assert strategy.root.children == children_nodes


def test_evaluate_node() -> None:
    """Test the evaluate_node method."""
    llm = FakeListChatModel(
        responses=["Explanation: Good trajectory. Correctness score: 8"]
    )
    strategy = LATSMathStrategy(llm=llm)

    root = strategy.initialize()
    child1 = Node(
        state=ReActOutput(
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
        state=ReActOutput(
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

    values = strategy.evaluate_node(root, question, examples, prompt, {})

    assert len(values) == 1  # Only one non-terminal child.
    assert "explanation" in values[0]
    assert "value" in values[0]
    assert values[0]["explanation"] == "Good trajectory."
    assert values[0]["value"] == 0.8  # 8 / 10

    assert child1.value == 0.8
    assert child2.value == 0  # Terminal node, value not updated.

    # Test caching.
    strategy.cache_values = True
    cached_values = strategy.evaluate_node(root, question, examples, prompt, {})
    assert cached_values == values

    # Test with empty reflection_map.
    strategy.reflection_map = []
    empty_reflection_values = strategy.evaluate_node(
        root, question, examples, prompt, {}
    )
    assert empty_reflection_values == values


def test_simulate_node() -> None:
    """Test the simulate_node method."""
    responses = [
        # "I need to search for the capital of France",
        # "Search[capital of France]",
        # "I need to search for the capital of France",
        # "Search[capital of France]",
        # "The trajectory provided is completely incorrect as the observation received does not relate to the search query at all, indicating that the search term might have been mistyped or confused",
        # "The search results did not return the information needed",
        # "Search[capital of France]\nObservation 2: The capital of France is Paris, known for its art, fashion, gastronomy, and culture",
        # "The search did not return relevant information",
        # "Search[capital of France Wikipedia]\nObservation 2: The capital of France is Paris, the largest city in France and its capital since the 4th century",
        # "The trajectory provided is incorrect because the environmental observation does not relate to the question asked",
        # "This trajectory is incorrect as it did not provide any relevant information regarding the capital of France",
        # "There seems to be an issue with the search results",
        # "Search[similar entities to the capital of France]\nObservation 3: Similar: [Paris, Marseille, Lyon, Toulouse, Lille]\nThought 4: The capital of France is Paris",
        # "The search results seem to be incorrect",
        # "Search[capital of France]\nObservation 3: The capital of France is Paris",
        # "The trajectory is incorrect as the observations did not provide any relevant information related to the question",
        # "This trajectory is incorrect as the focus should have been on verifying the information related to the capital of France, rather than repeatedly trying the same search query that does not provide the desired information",
    ]

    qa_strategy = LATSMathStrategy(
        llm=FakeListChatModel(responses=responses), depth_limit=3, n_samples=2
    )
    root_node = qa_strategy.initialize()

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

    reward, final_node, simulation_results = qa_strategy.simulate_node(
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

    assert isinstance(reward, float)
    assert isinstance(final_node, Node)
    assert isinstance(simulation_results, list)

    assert final_node.depth <= qa_strategy.depth_limit

    assert len(simulation_results) > 0

    assert -1 <= reward <= 1


def test_backpropagate_node() -> None:
    """Test the backpropagate_node method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm)

    # Create a simple tree structure.
    root = Node(state={})
    child = Node(state={}, parent=root)
    grandchild = Node(state={}, parent=child)
    grandchild.is_terminal = True

    # Test backpropagation for a successful terminal node.
    grandchild.reward = 1
    strategy.backpropagate_node(grandchild, 1.0)

    assert root.visits == 1
    assert child.visits == 1
    assert grandchild.visits == 1
    assert root.value == 1.0
    assert child.value == 1.0
    assert grandchild.value == 1.0

    # Test backpropagation for a failed terminal node.
    grandchild.reward = 0
    strategy.backpropagate_node(grandchild, 1.0)

    assert root.visits == 2
    assert child.visits == 2
    assert grandchild.visits == 2
    assert root.value == 1.0
    assert child.value == 1.0
    assert grandchild.value == 0.0

    # Test backpropagation for a non-terminal node.
    child.is_terminal = False
    strategy.backpropagate_node(child, 0.5)

    assert root.visits == 3
    assert child.visits == 3
    assert root.value == 5 / 6
    assert child.value == 5 / 6


def test_halting_condition() -> None:
    """Test the halting_condition method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm)

    # Test with a terminal node and reward of 1.
    terminal_node = Node(state={})
    terminal_node.is_terminal = True
    terminal_node.reward = 1
    assert strategy.halting_condition(terminal_node) is True

    # Test with a non-terminal node.
    non_terminal_node = Node(state={})
    assert strategy.halting_condition(non_terminal_node) is False

    # Test with a terminal node but reward is not 1.
    incorrect_terminal_node = Node(state={})
    incorrect_terminal_node.is_terminal = True
    incorrect_terminal_node.reward = 0
    assert strategy.halting_condition(incorrect_terminal_node) is False


def test_reflect_condition() -> None:
    """Test the reflect_condition method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm, max_unique=3, max_reflections=5)

    # Test when there are fewer unique trajectories than reflections
    strategy.failed_trajectories = [
        {"trajectory": f"t{i}", "final_answer": "answer"} for i in range(2)
    ]
    strategy.reflection_map = {}
    assert strategy.reflect_condition() is True

    # Test when there are more unique trajectories than reflections but less than max_reflections
    strategy.failed_trajectories = [
        {"trajectory": f"t{i}", "final_answer": f"answer{i}"} for i in range(4)
    ]
    strategy.reflection_map = {"r1": "reflection1"}
    assert strategy.reflect_condition() is True

    # Test when there are max_reflections unique trajectories
    strategy.failed_trajectories = [
        {"trajectory": f"t{i}", "final_answer": "answer"} for i in range(5)
    ]
    strategy.reflection_map = {
        "r1": "reflection1",
        "r2": "reflection2",
        "r3": "reflection3",
        "r4": "reflection4",
    }
    assert strategy.reflect_condition() is False


def test_reflect() -> None:
    """Test the reflect method."""
    llm = FakeListChatModel(responses=["Reflection 1", "Reflection 2"])
    strategy = LATSMathStrategy(llm=llm, max_unique=2)

    strategy.failed_trajectories = [
        {"trajectory": "Failed trajectory 1", "final_answer": "Incorrect answer 1"},
        {"trajectory": "Failed trajectory 2", "final_answer": "Incorrect answer 2"},
        {
            "trajectory": "Failed trajectory 1",
            "final_answer": "Incorrect answer 1",
        },  # Duplicate, should be ignored
    ]

    question = "What is the capital of France?"
    examples = "Example 1\nExample 2"
    prompt = "Reflect on the failed trajectory"
    additional_keys = {"key": "value"}

    reflections = strategy.reflect(question, examples, prompt, additional_keys)

    assert len(reflections) == 2
    assert reflections[0]["trajectory"] == "Failed trajectory 1"
    assert reflections[0]["reflection"] == "Reflection 1"
    assert reflections[1]["trajectory"] == "Failed trajectory 2"
    assert reflections[1]["reflection"] == "Reflection 2"

    assert strategy.reflection_map == reflections


def test_reset() -> None:
    """Test the reset method."""
    llm = FakeListChatModel(responses=[])
    strategy = LATSMathStrategy(llm=llm)

    strategy.root = "some_root"
    strategy.reflection_map = ["reflection1", "reflection2"]
    strategy.value_cache = {"value1": "value2"}
    strategy.failed_trajectories = ["trajectory1", "trajectory2"]

    # Call reset.
    strategy.reset()

    # Check if the state has been reset.
    assert strategy.root is None
    assert strategy.failed_trajectories == []
    assert strategy.reflection_map == []
    assert strategy.value_cache == {}


def test_instantiate_strategies() -> None:
    """Test the instantiation of various LATS Math strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = LATSMathStrategy(llm=llm)
    svamp_strategy = LATSSVAMPStrategy(llm=llm)
    tabmwp_strategy = LATSTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, LATSMathStrategy)
    assert isinstance(svamp_strategy, LATSSVAMPStrategy)
    assert isinstance(tabmwp_strategy, LATSTabMWPStrategy)

