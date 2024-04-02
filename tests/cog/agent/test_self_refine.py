"""Unit tests for Self-Refine."""

from discussion_agents.cog.agent.self_refine import SelfRefineAgent
from discussion_agents.cog.modules.memory.self_refine import SelfRefineMemory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models.fake import FakeListChatModel


def test_init() -> None:
    """Test initialization."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['1']))
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, SelfRefineMemory)

    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['1']), memory=SelfRefineMemory(solution=["solution #1"]))
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, SelfRefineMemory)
    assert agent.memory.solution[0] == "solution #1"


def test_reset() -> None:
    """Test reset."""
    agent = SelfRefineAgent(
        llm=FakeListChatModel(responses=['response']), 
        memory=SelfRefineMemory(solution=["solution1"], feedback=['feedback1'])
    )
    assert agent.memory.solution != []
    assert agent.memory.feedback != []
    agent.reset()
    assert agent.memory.solution == []
    assert agent.memory.feedback == []


def test_retrieve() -> None:
    """Test retrieve."""
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=['response']))
    agent.memory.add_memories('solution1', 'feedback1')
    retrieved_memory = agent.retrieve()
    assert retrieved_memory['solution'] == ['solution1']
    assert retrieved_memory['feedback'] == ['feedback1']


def test_generate() -> None:
    """Test generate."""
    question = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    
    gt_out = ['def solution():\n    """A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"""\n    blue_fiber = 2\n    white_fiber = blue_fiber / 2\n    total_bolts = blue_fiber + white_fiber\n    result = total_bolts\n    return result']    
    responses = [
        'def solution():\n    """A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"""\n    blue_fiber = 2\n    white_fiber = blue_fiber / 2\n    total_bolts = blue_fiber + white_fiber\n    result = total_bolts\n    return result',
        'There is no error in the code! It is correct.'
    ]
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=responses))
    out = agent.generate(question=question)
    assert out == gt_out

    # Test with refinement.
    question = "Billy is buying some candy with $10 his father gave him. The candy costs $1.5 a pound. After buying candy, he takes half his change and spends it on gumballs, which cost $.05 each. If he bought 40 gumballs, how many pounds of candy did he buy?"
    
    gt_out = [
        'def solution():\n    """Billy is buying some candy with $10 his father gave him. The candy costs $1.5 a pound. After buying candy, he takes half his change and spends it on gumballs, which cost $.05 each. If he bought 40 gumballs, how many pounds of candy did he buy?"""\n    money_initial = 10\n    candy_cost = 1.5\n    gumball_cost = 0.05\n    gumballs_bought = 40\n\n    candy_weight = (money_initial / 2) / candy_cost\n    return candy_weight\n\nprint(solution())', 
        'The error in the code is in the calculation of the candy_weight. The candy_weight should be calculated based on the remaining money after buying gumballs, not half of the initial money. The correct calculation should be as follows:\n\n```python\ncandy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n```\n\nThe corrected code would be:\n\n```python\ndef solution():\n    money_initial = 10\n    candy_cost = 1.5\n    gumball_cost = 0.05\n    gumballs_bought = 40\n\n    candy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n    return candy_weight\n\nprint(solution())\n```\n\nThank you for pointing out the error!', 
        'The error in the code is in the calculation of the candy_weight. The candy_weight should be calculated based on the remaining money after buying gumballs, not half of the initial money. The correct calculation should be as follows:\n\n```python\ncandy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n```\n\nThe corrected code would be:\n\n```python\ndef solution():\n    money_initial = 10\n    candy_cost = 1.5\n    gumball_cost = 0.05\n    gumballs_bought = 40\n\n    candy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n    return candy_weight\n\nprint(solution())\n```\n\nThank you for pointing out the error!'
    ]
    responses = [
        'def solution():\n    """Billy is buying some candy with $10 his father gave him. The candy costs $1.5 a pound. After buying candy, he takes half his change and spends it on gumballs, which cost $.05 each. If he bought 40 gumballs, how many pounds of candy did he buy?"""\n    money_initial = 10\n    candy_cost = 1.5\n    gumball_cost = 0.05\n    gumballs_bought = 40\n\n    candy_weight = (money_initial / 2) / candy_cost\n    return candy_weight\n\nprint(solution())',
        'The error in the code is in the calculation of the candy_weight. The candy_weight should be calculated based on the remaining money after buying gumballs, not half of the initial money. The correct calculation should be as follows:\n\n```python\ncandy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n```\n\nThe corrected code would be:\n\n```python\ndef solution():\n    money_initial = 10\n    candy_cost = 1.5\n    gumball_cost = 0.05\n    gumballs_bought = 40\n\n    candy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n    return candy_weight\n\nprint(solution())\n```\n\nThank you for pointing out the error!',
        'The error in the code is in the calculation of the candy_weight. The candy_weight should be calculated based on the remaining money after buying gumballs, not half of the initial money. The correct calculation should be as follows:\n\n```python\ncandy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n```\n\nThe corrected code would be:\n\n```python\ndef solution():\n    money_initial = 10\n    candy_cost = 1.5\n    gumball_cost = 0.05\n    gumballs_bought = 40\n\n    candy_weight = (money_initial - (gumballs_bought * gumball_cost)) / candy_cost\n    return candy_weight\n\nprint(solution())\n```\n\nThank you for pointing out the error!',
        'It is correct.'
    ]
    agent = SelfRefineAgent(llm=FakeListChatModel(responses=responses))
    out = agent.generate(question=question)
    assert out == gt_out