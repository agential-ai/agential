"""Unit tests for CRITIC."""

from unittest.mock import MagicMock

import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.critic import CriticAgent
from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    GSM8K_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT,
    HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST,
)


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    search = MagicMock(spec=GoogleSearchAPIWrapper)
    agent = CriticAgent(llm=llm, mode="qa", search=search)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(search, GoogleSearchAPIWrapper)

    with pytest.raises(ValueError):
        agent = CriticAgent(llm=llm, mode="qa", search=None)


def test_generate() -> None:
    """Test generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    # Test "qa" mode without search tool.
    responses = [
        "Let's break it down step by step. The person described is a former kickboxer who was once considered the best in the world but has been involved in controversies and crimes of violence. Based on this description, the person in question is likely Badr Hari. So the answer is: Badr Hari.",
        'The question asks for the name of a person meeting specific criteria, and the answer "Badr Hari" is a name that fits the description provided. So it\'s plausible.',
        "the most possible answer: Let's think step by step. The person described is a former kickboxer who was once considered the best in the world but has been involved in controversies and crimes of violence. Based on this description, the person in question is likely Badr Hari. So the answer is: Badr Hari.",
    ]
    search = MagicMock(spec=GoogleSearchAPIWrapper)
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), mode="qa", search=search
    )
    out = agent.generate(question=question, use_search_tool=False)
    assert isinstance(out, list)
    assert len(out) == 2

    # Test "qa" mode with search tool.
    responses = [
        "Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        'The question asks for a kick boxer who was once considered the best in the world but has been involved in controversies and crimes. The answer "Badr Hari" is a plausible response.\n\n2. Truthfulness:\n\nLet\'s search the question in google:\n\n> Search Query: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, fighting out of Mike\'s Gym in Oostzaan. Hari has been a prominent figure in the world of kickboxing and considered one of the best kickboxers in the world.\n\nThe evidence supports that Badr Hari was once considered one of the best kickboxers in the world but has been involved in controversies and crimes.\n\nTherefore, the proposed answer is both plausible and truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\nHere\'s the most possible answer: Let\'s think step by step. The kick boxer who was once considered the best in the world, involved in controversies and crimes is Badr Hari. So the answer is: Badr Hari.',
        "The evidence does not provide specific information about the kick boxer in question.\n\nLet's search the proposed answer in google:\n\n> Search Query: Badr Hari kickboxing controversies\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch super heavyweight kickboxer from Amsterdam, fighting out of Mike's Gym in Oostzaan.\n\nThe evidence confirms that Badr Hari is a kick boxer known for controversies and unsportsmanlike conduct, supporting the proposed answer.\n\nOverall, the proposed answer is plausible and truthful based on the evidence found.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his \"unsportsmanlike conducts\" in the sport and crimes of violence outside of the ring\nHere's the most possible answer: Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        'The evidence supports the fact that Badr Hari has been involved in controversies and crimes related to kickboxing and outside of the ring, making him a suitable answer to the question. \n\nTherefore, the proposed answer "Badr Hari" is correct and aligns with the information available.',
        "the most possible answer: Let's think step by step. The kick boxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
    ]
    search = MagicMock(spec=GoogleSearchAPIWrapper)
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), mode="qa", search=search
    )
    out = agent.generate(question=question)
    assert isinstance(out, list)
    assert len(out) == 4

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # Test "math" mode without interpreter tool.
    responses = [
        "total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg",
        "Question: Jacob is going on a road trip. His car gets 25 miles per gallon of gas. If the trip is 400 miles long, how many gallons of gas does he need?\n\n```python\nmiles_per_gallon = 25\ntrip_length = 400\ngallons_needed = trip_length / miles_per_gallon\nanswer = gallons_needed\n```\n\nWhat do you think of the above code?\n\n1. The calculation of gallons needed based on the miles per gallon and trip length is correct.\n2. The answer should be a positive number, 16 > 0, which is reasonable.\n3. The code is clear and concise, and it correctly calculates the gallons of gas needed for the trip.",
        "Question: Tommy is making 12 loaves of bread. He needs 4 pounds of flour per loaf. A 10-pound bag of flour costs $10 and a 12-pound bag costs $13. When he is done making his bread, he has no use for flour and so he will throw away whatever is left. How much does he spend on flour if he buys the cheapest flour to get enough?\n```python\nnum_of_loaves = 12\npounds_of_flour_per_loaf = 4\ntotal_pounds_needed = num_of_loaves * pounds_of_flour_per_loaf\n\ncost_10_pound_bag = 10\ncost_12_pound_bag = 13\n\n# Calculate the number of bags needed\nbags_10_pound = total_pounds_needed // 10\nbags_12_pound = total_pounds_needed // 12\n\n# Calculate the cost for each type of bag\ntotal_cost_10_pound = bags_10_pound * cost_10_pound_bag\ntotal_cost_12_pound = bags_12_pound * cost_12_pound_bag\n\n# Choose the cheaper option\ntotal_cost = min(total_cost_10_pound, total_cost_12_pound)\n\nanswer = total_cost\n```\n\nIn this revised code:\n- The total pounds of flour needed is correctly calculated.\n- The number of bags needed is calculated by integer division to ensure whole bags are considered.\n- The total cost for each type of bag is calculated.\n- The cheaper option between the 10-pound and 12-pound bag is chosen as the total cost.",
        "Question: Janet hires six employees. Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour. Janet has to pay 10% of her workers' salaries in FICA taxes. If everyone works 25 days a month and 8 hours a day, how much does Janet owe total for their wages and taxes for one month?\n```python\nnum_of_warehouse_workers = 4\nnum_of_managers = 2\nwage_of_warehouse_workers = 15\nwage_of_managers = 20\nnum_of_days = 25\nnum_of_hours = 8\ntotal_hours = num_of_days * num_of_hours\ntotal_wage = num_of_warehouse_workers * wage_of_warehouse_workers * total_hours + num_of_managers * wage_of_managers * total_hours\nanswer = total_wage * 1.1\n```\n\nWhat's the problem with the above code?\n\n1. The code correctly calculates the total amount owed for wages and taxes.\n\n2. The code could be improved by adding comments for better readability and organization. \n\nHere's the same code with added comments:\n\n```python\n# Define the number of warehouse workers and managers\nnum_of_warehouse_workers = 4\nnum_of_managers = 2\n\n# Define the wages for warehouse workers and managers\nwage_of_warehouse_workers = 15\nwage_of_managers = 20\n\n# Calculate the total number of hours worked in a month\nnum_of_days = 25\nnum_of_hours = 8\ntotal_hours = num_of_days * num_of_hours\n\n# Calculate the total wages for the workers\ntotal_wage = num_of_warehouse_workers * wage_of_warehouse_workers * total_hours + num_of_managers * wage_of_managers * total_hours\n\n# Calculate the total FICA taxes owed (10% of total wages)\ntotal_fica_tax = total_wage * 0.1\n\n# Calculate the total amount owed (wages + taxes)\ntotal_owed = total_wage + total_fica_tax\nanswer = total_owed\n``` \n\nThis code snippet is correct and calculates the total amount Janet owes for the wages and taxes for one month.",
        'Question: In a school, the average weight of boys is 60 kg, and the average weight of girls is 45 kg. If the average weight of all students in the school is 50 kg, what is the ratio of boys to girls in the school?\n```python\navg_weight_boys = 60\navg_weight_girls = 45\ntotal_avg_weight = 50\nratio_boys = (total_avg_weight - avg_weight_girls) / (avg_weight_boys - total_avg_weight)\nratio_girls = 1 - ratio_boys\nanswer = f"{ratio_boys}:{ratio_girls}"\n```\n\nWhat\'s the problem with the above code?\n\n1. The ratio of boys to girls should be in integer form, for example, 2:3, so the answer should be simplified.\n\n2. Let\'s check the code:\n\n> avg_weight_boys = 60\n> avg_weight_girls = 45\n> total_avg_weight = 50\n\nIt defines the average weights correctly.\n\n> ratio_boys = (total_avg_weight - avg_weight_girls) / (avg_weight_boys - total_avg_weight)\n> ratio_girls = 1 - ratio_boys\n\nThe calculation of ratios seems to be incorrect. The ratio of boys to girls is based on their weights, so we should consider the weights to find the correct ratio.\n\nHere\'s a better solution:\n```python\n# Given data\navg_weight_boys = 60\navg_weight_girls = 45\ntotal_avg_weight = 50\n\n# Let the number of boys be x and number of girls be y\n# We can set up the following equations based on the weights and average weight\n# x * 60 + y * 45 = (x + y) * 50\n# x + y = total_students\n\n# Solving the equations\ntotal_students = 1  # assume total students is 1 (for ratios)\ny = (60 - total_avg_weight) / (total_avg_weight - 45)  # number of girls\nx = total_students - y  # number of boys\n\n# Calculate the simplified ratio of boys to girls\n# Since the total_students is 1, we can simplify the ratio\nratio_boys = int(x)\nratio_girls = int(y)\n\nanswer = f"{ratio_boys}:{ratio_girls}"\n```\n\nThis solution correctly calculates the ratio of boys to girls based on their average weights and the total average weight.',
    ]
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), mode="math")
    out = agent.generate(
        question=question,
        use_interpreter_tool=False,
        examples=GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_GSM8K,
        critique_examples=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        critique_prompt=GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        max_interactions=2,
    )
    assert isinstance(out, list)
    assert len(out) == 2

    # Test "math" mode with interpreter tool.
    responses = [
        "total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\nanswer = sold_eggs * dollars_per_egg",
        "The problem with the above code is that the calculation for the number of eggs sold is incorrect. \n\nLet's correct the code:\n\n```python\ntotal_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\ndollars_per_egg = 2\n\n# Make sure the number of sold eggs is non-negative\nif sold_eggs < 0:\n    sold_eggs = 0\n\nanswer = sold_eggs * dollars_per_egg\n```",
        "total_eggs = 16\neaten_eggs = 3\nbaked_eggs = 4933828\nsold_eggs = total_eggs - eaten_eggs - baked_eggs\n\n# Check if the number of sold eggs is negative and correct it\nif sold_eggs < 0:\n    sold_eggs = 0\n\ndollars_per_egg = 2\ntotal_money_made = sold_eggs * dollars_per_egg\n\nanswer = total_money_made",
        "It is correct.",
    ]
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), mode="math")
    out = agent.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_GSM8K,
        critique_examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC,
        critique_prompt=CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    )
    assert isinstance(out, list)
    assert len(out) == 2

    question = """from typing import List 

    def has_close_elements(numbers: List[float], threshold: float) -> bool: 
        \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold. 
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False 
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True
        \"\"\"
    """

    # Test "code" mode without interpreter tool.
    responses = [
        'from typing import List \n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool: \n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold. \n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False \n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True\n    """\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False',
        "It is correct.",
    ]
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), mode="code")
    out = agent.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_HUMANEVAL,
        critique_examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        critique_prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
        use_interpreter_tool=False,
        test_prompt=CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL,
        test_examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST,
    )
    assert isinstance(out, list)
    assert len(out) == 1

    # Test "code" mode with interpreter tool.
    responses = [
        'from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """\n    Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5) \n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) \n    True\n    """\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False',
        'assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False, "Test failed: has_close_elements([1.0, 2.0, 3.0], 0.5) should return False"\nassert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True, "Test failed: has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) should return True"',
        "It is correct.",
    ]
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), mode="code")
    out = agent.generate(
        question=question,
        examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_HUMANEVAL,
        critique_examples=HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
        critique_prompt=CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
        use_interpreter_tool=True,
        test_prompt=CRITIC_POT_INSTRUCTION_TEST_HUMANEVAL,
        test_examples=HUMANEVAL_FEWSHOT_EXAMPLES_POT_TEST,
    )
    assert isinstance(out, list)
    assert len(out) == 1
