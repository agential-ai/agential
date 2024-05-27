"""Unit tests for CRITIC."""

from unittest.mock import MagicMock

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.critic import CriticAgent
from agential.cog.prompts.critic import (
    CRITIC_CRITIQUE_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_INSTRUCTION_MBPP,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_HUMANEVAL,
    CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_MBPP,
    CRITIC_POT_INSTRUCTION_GSM8K,
    CRITIC_POT_INSTRUCTION_HUMANEVAL,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC,
    GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC,
    HUMANEVAL_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
    MBPP_FEWSHOT_EXAMPLES_CRITIC,
    MBPP_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
)
from agential.cog.prompts.benchmarks.gsm8k import GSM8K_FEWSHOT_EXAMPLES_POT
from agential.cog.prompts.benchmarks.humaneval import HUMANEVAL_FEWSHOT_EXAMPLES_POT


def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    search = MagicMock(spec=GoogleSerperAPIWrapper)
    agent = CriticAgent(llm=llm, mode={"qa": "hotpotqa"}, search=search)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(search, GoogleSerperAPIWrapper)


def test_generate() -> None:
    """Test generate method."""
    question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    # Test "qa" mode without search tool.
    responses = [
        "Let's think step by step. The person described in the question is Mike Tyson, who was once considered the best kickboxer in the world but has been involved in controversies and crimes of violence. So the answer is: Mike Tyson.",
        'The question specifies that the individual was once considered the best kickboxer in the world, however, Mike Tyson is not a kickboxer, he is a former professional boxer. So the answer is not plausible.\n\n2. Truthfulness:\n\nLet\'s search the question in google:\n\n> Search Query: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\n> Evidence: There is no relevant information available in the search results to confirm the accuracy of the proposed answer.\n\nGiven that the proposed answer incorrectly identifies Mike Tyson as a kickboxer, it is not truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\nHere\'s the most possible answer: The individual described in the question is not Mike Tyson, as he is a former professional boxer, not a kickboxer. Without further information, it is not possible to determine the correct answer.',
        'There is no specific evidence found for this question.\n\nLet\'s search the proposed answer in google:\n\n> Search Query: Mike Tyson kickboxing\n> Evidence: [Mike Tyson - Wikipedia] Tyson made his professional kickboxing debut on May 20, 2006, against Jonathan Paredes in a three-round exhibition. Tyson knocked Paredes down in the first round with a right hook.\n\nThe evidence shows that Mike Tyson did participate in kickboxing, but he is primarily known as a boxer, not a kickboxer. \n\nTherefore, the proposed answer is not accurate in terms of the sport, and it should be revised to reflect the correct sport. \n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the revised answer: The person described in the question is Mike Tyson, who was once considered one of the best professional boxers in the world but has been involved in controversies and crimes of violence. So the answer is: Mike Tyson.',
        '\n> Evidence: [Mike Tyson - Wikipedia] Mike Tyson is not primarily known for kickboxing. He is a former professional boxer who competed from 1985 to 2005.\n\nThe evidence confirms that Mike Tyson is not a kickboxer but a former professional boxer.\n\nTherefore, the proposed answer is incorrect and needs revision.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the most possible answer: Let\'s think step by step. The individual described in the question is not Mike Tyson, who is a former professional boxer, not a kickboxer. The correct answer needs further research to identify the kickboxer matching the description provided.',
        'the most possible answer: The individual described in the question is not Mike Tyson, as he is a former professional boxer, not a kickboxer. Unfortunately, without further information or evidence, it is not possible to determine the correct answer to this question.',
    ]
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), mode={"qa": "hotpotqa"}
    )
    out = agent.generate(question=question, use_tool=False)
    assert isinstance(out, list)
    assert len(out) == 2

    # Test "qa" mode with search tool.
    search = MagicMock(spec=GoogleSerperAPIWrapper)
    responses = [
        "Let's break it down step by step. The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.",
        'The question asks for a kickboxer who fits the description provided, and the answer "Badr Hari" is a plausible response.\n\n2. Truthfulness:\n\nLet\'s search the question in Google:\n\n> Search Query: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch super heavyweight kickboxer from the Netherlands, fighting out of Mike\'s Gym in Oostzaan. He is a former K-1 Heavyweight Champion (2007-2008) and It\'s Showtime Heavyweight Champion (2009-2010).\n\nThe evidence confirms that Badr Hari fits the description provided in the question.\n\nOverall, the proposed answer is both plausible and truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the most possible answer: Let\'s break it down step by step. The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.',
        'The evidence does not provide any relevant information about the question.\n\nLet\'s search the proposed answer in Google:\n\n> Search Query: Badr Hari kickboxer controversies\n> Evidence: [Badr Hari - Wikipedia] Badr Hari is a Moroccan-Dutch kickboxer. He is a former K-1 Heavyweight champion, It\'s worth noting that Hari has been involved in several controversies, including unsportsmanlike conduct and criminal charges.\n\nThe evidence supports the claim that Badr Hari fits the description provided in the question.\n\nOverall, the proposed answer is both plausible and truthful.\n\nQuestion: Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?\nHere\'s the most possible answer: Let\'s break it down step by step. The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.',
        'the most possible answer: The kickboxer who fits this description is Badr Hari. So the answer is: Badr Hari.',
    ]
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), 
        mode={"qa": "hotpotqa"},
        search=search
    )
    out = agent.generate(question=question, use_tool=True)
    assert isinstance(out, list)
    assert len(out) == 3

    # Test "math" mode without code interpreter tool.
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    responses = [
        'eggs_laid_per_day = 16\neggs_used_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_used_for_breakfast - eggs_used_for_muffins\nmoney_earned_per_egg = 2\nmoney_earned_per_day = eggs_remaining * money_earned_per_egg\nanswer = money_earned_per_day',
        "1. The amount of money earned per day should be a positive number, 0 < 10 million, it's reasonable.\n\n2. Let's check the code:\n\n> eggs_laid_per_day = 16\n> eggs_used_for_breakfast = 3\n> eggs_used_for_muffins = 4933828\n\nThis correctly defines the number of eggs laid per day, used for breakfast, and used for muffins.\n\n> eggs_remaining = eggs_laid_per_day - eggs_used_for_breakfast - eggs_used_for_muffins\n\nThis calculates the number of eggs remaining after breakfast and muffins are accounted for, which is correct.\n\n> money_earned_per_egg = 2\n> money_earned_per_day = eggs_remaining * money_earned_per_egg\n\nThis calculates the amount of money earned per day at the farmers' market based on the remaining eggs and the price per egg, which is correct.\n\nOverall, the code is correct and provides the amount of money Janet makes every day at the farmers' market.",
        "# Given values\neggs_laid_per_day = 16\neggs_used_for_breakfast = 3\neggs_used_for_muffins = 4933828\nmoney_earned_per_egg = 2\n\n# Calculate the number of eggs remaining after breakfast and muffins\neggs_remaining = eggs_laid_per_day - eggs_used_for_breakfast - eggs_used_for_muffins\n\n# Calculate the money earned per day at the farmers' market\nmoney_earned_per_day = eggs_remaining * money_earned_per_egg\n\nanswer = money_earned_per_day",
        "The above code has a logical error in calculating the number of eggs remaining after breakfast and muffins. The subtraction operation is incorrect because it should be subtracting the eggs used for breakfast and muffins from the total eggs laid per day. \n\nHere's the corrected calculation:\n```python\n# Calculate the number of eggs remaining after breakfast and muffins\neggs_remaining = eggs_laid_per_day - eggs_used_for_breakfast - eggs_used_for_muffins\n\n# Calculate the money earned per day at the farmers' market\nmoney_earned_per_day = eggs_remaining * money_earned_per_egg\n\nanswer = money_earned_per_day\n``` \n\nAfter this correction, the code will accurately calculate the amount of money Janet makes every day at the farmers' market.",
        "# Given values\neggs_laid_per_day = 16\neggs_used_for_breakfast = 3\neggs_used_for_muffins = 4933828\nmoney_earned_per_egg = 2\n\n# Calculate the total eggs available for sale\neggs_remaining = eggs_laid_per_day - eggs_used_for_breakfast - eggs_used_for_muffins\n\n# Ensure the number of eggs remaining is not negative\nif eggs_remaining < 0:\n    eggs_remaining = 0\n\n# Calculate the money earned per day at the farmers' market\nmoney_earned_per_day = eggs_remaining * money_earned_per_egg\n\nanswer = money_earned_per_day\n```",
        'The code is correct and there is no issue with it.',
        "# Given values\neggs_laid_per_day = 16\neggs_used_for_breakfast = 3\neggs_used_for_muffins = 4933828\nmoney_earned_per_egg = 2\n\n# Calculate the total eggs available for sale\neggs_remaining = eggs_laid_per_day - eggs_used_for_breakfast - eggs_used_for_muffins\n\n# Ensure the number of eggs remaining is not negative\neggs_remaining = max(eggs_remaining, 0)\n\n# Calculate the money earned per day at the farmers' market\nmoney_earned_per_day = eggs_remaining * money_earned_per_egg\n\nanswer = money_earned_per_day\n```",    
    ]
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), 
        mode={"math": "gsm8k"},
    )
    out = agent.generate(question=question, use_tool=False)
    assert isinstance(out, list)
    assert len(out) == 7

    # Test "math" mode with code interpreter tool.
    responses = [
        'eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day',
        "1. The amount of money made every day at the farmers' market should be a positive number, -9867630 < 0, which is not reasonable.\n\n2. Let's analyze the code:\n\n- `eggs_laid_per_day = 16`: This variable correctly represents the number of eggs Janet's ducks lay per day.\n- `eggs_for_breakfast = 3`: This variable correctly represents the number of eggs Janet eats for breakfast every morning.\n- `eggs_used_for_muffins = 4933828`: This variable seems to be incorrectly set to a very large number, which doesn't align with the scenario of baking muffins daily. This is likely a mistake.\n- `eggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins`: This calculation is incorrect due to the extremely high value of `eggs_used_for_muffins`, leading to a negative value for `eggs_remaining`.\n- `money_made_per_day = eggs_remaining * 2`: This calculation is based on the incorrect value of `eggs_remaining`, leading to the negative result.\n\nTo correct the code and calculate the amount of money Janet makes every day at the farmers' market, we need to adjust the calculation of `eggs_remaining` based on a realistic scenario. Let's assume that Janet uses a certain number of eggs for muffins daily, and the rest are sold at the market.\n\nHere's a revised solution:\n\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 6  # Let's assume Janet uses 6 eggs for muffins daily\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\nif eggs_remaining < 0:\n    eggs_remaining = 0  # Ensure that the number of remaining eggs is non-negative\n\nmoney_made_per_day = eggs_remaining * 2\nanswer = money_made_per_day\n``` \n\nIn this revised solution, we have adjusted the number of eggs used for muffins to a more realistic value (6 eggs in this case) and made sure that the calculation of `eggs_remaining` does not result in a negative value. The final amount of money made every day at the farmers' market is then calculated based on the corrected number of remaining eggs.",
        "# Given data\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 12  # Assuming a more reasonable number of eggs used for muffins\negg_price = 2  # Price per fresh duck egg\n\n# Calculate the number of eggs remaining to be sold\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\n\n# Calculate the money made per day at the farmers' market\nmoney_made_per_day = eggs_remaining * egg_price\n\nanswer = money_made_per_day",
        "There are a few issues with the code provided:\n\n1. The calculation for the number of eggs used for muffins seems incorrect. It states that 12 eggs are used for muffins, but the total number of eggs laid per day is 16. This would mean that only 4 eggs are left for sale, which doesn't align with the final calculation result of $2.\n\n2. The variable names could be more descriptive to enhance code readability.\n\n3. It's unclear what the variable `4933828` represents in the context of the problem. It seems like it might be a typo or irrelevant to the calculation.\n\nHere's an improved version of the code:\n\n```python\n# Given data\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_for_muffins = 1  # Assuming 1 egg is used for muffins, leaving 12 for sale\negg_price = 2  # Price per fresh duck egg\n\n# Calculate the number of eggs remaining to be sold\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_used_for_muffins\n\n# Calculate the money made per day at the farmers' market\nmoney_made_per_day = eggs_remaining * egg_price\n\nanswer = money_made_per_day\n```\n\nThis revised code assumes that only 1 egg is used for muffins, leaving 12 eggs for sale, which results in a daily income of $24 at the farmers' market.",
        "# Given data\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4  # Assuming 4 eggs are used for muffins\negg_price = 2  # Price per fresh duck egg\n\n# Calculate the number of eggs remaining to be sold\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\n\n# Calculate the money made per day at the farmers' market\nmoney_made_per_day = eggs_remaining * egg_price\n\nanswer = money_made_per_day\n```",
        "There seems to be a mistake in the calculation of the number of eggs used for muffins. The question states that Janet bakes muffins for her friends every day with some of the eggs, but the code assumes that she uses 4 eggs for muffins. Since the number of eggs used for muffins is not explicitly provided in the question, we should not make assumptions about it.\n\nTo correct this, we can adjust the code to use the actual number of eggs used for muffins as stated in the question. Here's an updated solution:\n\n```python\n# Given data\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 0  # Assuming no eggs are used for muffins\negg_price = 2  # Price per fresh duck egg\n\n# Calculate the number of eggs remaining to be sold\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\n\n# Calculate the money made per day at the farmers' market\nmoney_made_per_day = eggs_remaining * egg_price\n\nanswer = money_made_per_day\n```\n\nThis updated code reflects the fact that no eggs are used for muffins, as specified in the question.",
        "# Given data\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828  # Janet bakes muffins with 4933828 eggs for her friends every day\negg_price = 2  # Price per fresh duck egg\n\n# Calculate the number of eggs remaining to be sold\neggs_remaining = eggs_laid_per_day - eggs_for_breakfast - eggs_for_muffins\n\n# Calculate the money made per day at the farmers' market\nmoney_made_per_day = eggs_remaining * egg_price\n\nanswer = money_made_per_day\n```",
    ]
    agent = CriticAgent(
        llm=FakeListChatModel(responses=responses), 
        mode={"math": "gsm8k"},
    )
    out = agent.generate(question=question, use_tool=True)
    assert isinstance(out, list)
    assert len(out) == 7

    # Test "code" mode without code interpreter tool.

    # Test "code" mode with code interpreter tool.

    # Test "code" mode with code interpreter tool and unit tests.