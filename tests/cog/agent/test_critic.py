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
    search = MagicMock(spec=GoogleSearchAPIWrapper)
    agent = CriticAgent(llm=llm, mode={"qa": "hotpotqa"}, search=search)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(search, GoogleSearchAPIWrapper)


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
    out = agent.generate(question=question, use_tool=False)
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
        "The code is correct.",
    ]
    agent = CriticAgent(llm=FakeListChatModel(responses=responses), mode="math")
    out = agent.generate(
        question=question,
        use_tool=False,
        examples=GSM8K_FEWSHOT_EXAMPLES_POT,
        prompt=CRITIC_POT_INSTRUCTION_GSM8K,
        critique_examples=GSM8K_FEWSHOT_EXAMPLES_CRITIC_NO_TOOL,
        critique_prompt=CRITIC_CRITIQUE_NO_TOOL_INSTRUCTION_GSM8K,
        max_interactions=2,
    )
    assert isinstance(out, list)
    assert len(out) == 1

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

    # Test "code" mode for HumanEval without interpreter tool.
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
        use_tool=False,
        
        
    )
    assert isinstance(out, list)
    assert len(out) == 1

    # Test "code" mode for HumanEval with interpreter tool.
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
        use_tool=True,
        
        
    )
    assert isinstance(out, list)
    assert len(out) == 1

    tests = """assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False"""